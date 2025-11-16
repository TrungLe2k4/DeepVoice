// utils_dsp.js — Core DSP helpers for DeepVoice Guard (no external deps)
// All APIs are attached to global self.DSP (or window.DSP) for importScripts() usage.

(function (g) {
  "use strict";

  const DSP = {};

  /* ---------- Window ---------- */
  DSP.hanning = function (n) {
    const w = new Float32Array(n);
    const f = (2 * Math.PI) / (n - 1);
    for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos(f * i);
    return w;
  };

  /* ---------- FFT (radix-2, in-place, real->complex arrays) ---------- */
  // re, im are Float32Array length N (N power-of-two)
  DSP.fftRadix2 = function (re, im) {
    const n = re.length;
    let i = 0,
      j = 0;
    // bit-reversal permutation
    for (i = 1; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        let tr = re[i];
        re[i] = re[j];
        re[j] = tr;
        let ti = im[i];
        im[i] = im[j];
        im[j] = ti;
      }
    }
    // Cooley–Tukey
    for (let len = 2; len <= n; len <<= 1) {
      const ang = (-2 * Math.PI) / len;
      const wlen_r = Math.cos(ang),
        wlen_i = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let w_r = 1,
          w_i = 0;
        for (let j = 0; j < (len >> 1); j++) {
          const u_r = re[i + j],
            u_i = im[i + j];
          const v_r =
              re[i + j + (len >> 1)] * w_r -
              im[i + j + (len >> 1)] * w_i,
            v_i =
              re[i + j + (len >> 1)] * w_i +
              im[i + j + (len >> 1)] * w_r;
          re[i + j] = u_r + v_r;
          im[i + j] = u_i + v_i;
          re[i + j + (len >> 1)] = u_r - v_r;
          im[i + j + (len >> 1)] = u_i - v_i;
          // w *= wlen
          const nw_r = w_r * wlen_r - w_i * wlen_i;
          const nw_i = w_r * wlen_i + w_i * wlen_r;
          w_r = nw_r;
          w_i = nw_i;
        }
      }
    }
  };

  /* ---------- Magnitude Spectrum (real time-domain input) ---------- */
  DSP.magSpectrum = function (frame, win /* Float32Array | null */) {
    const N = frame.length;
    const re = new Float32Array(N);
    const im = new Float32Array(N);
    if (win) {
      for (let i = 0; i < N; i++) re[i] = frame[i] * win[i];
    } else {
      re.set(frame);
    }
    DSP.fftRadix2(re, im);
    const out = new Float32Array(N >> 1);
    for (let k = 0; k < out.length; k++) {
      const r = re[k],
        ii = im[k];
      out[k] = Math.hypot(r, ii);
    }
    return out;
  };

  /* ---------- Mel helpers ---------- */
  DSP.hzToMel = (hz) => 2595 * Math.log10(1 + hz / 700);
  DSP.melToHz = (mel) => 700 * (Math.pow(10, mel / 2595) - 1);

  /* ---------- Mel Filterbank (triangular) ---------- */
  DSP.melFilterbank = function (nfft, sr, nMels = 64, fmin = 50, fmax = sr / 2) {
    const mMin = DSP.hzToMel(fmin),
      mMax = DSP.hzToMel(fmax);
    const mPts = new Float32Array(nMels + 2);
    for (let i = 0; i < mPts.length; i++) {
      mPts[i] = mMin + (mMax - mMin) * (i / (nMels + 1));
    }
    const hz = new Float32Array(mPts.length);
    for (let i = 0; i < mPts.length; i++) hz[i] = DSP.melToHz(mPts[i]);

    const bins = new Int32Array(hz.length);
    for (let i = 0; i < hz.length; i++)
      bins[i] = Math.floor(((nfft + 1) * hz[i]) / sr);

    const fb = new Array(nMels);
    for (let m = 1; m <= nMels; m++) {
      const f = new Float32Array(nfft >> 1);
      for (let k = bins[m - 1]; k < bins[m]; k++) {
        if (k < f.length)
          f[k] =
            (k - bins[m - 1]) /
            Math.max(1, bins[m] - bins[m - 1]);
      }
      for (let k = bins[m]; k < bins[m + 1]; k++) {
        if (k < f.length)
          f[k] =
            (bins[m + 1] - k) /
            Math.max(1, bins[m + 1] - bins[m]);
      }
      fb[m - 1] = f;
    }
    return fb;
  };

  /* ---------- Linear Filterbank (LFCC) ---------- */
  DSP.linearFilterbank = function (
    nfft,
    sr,
    nBands = 40,
    fmin = 50,
    fmax = sr / 2
  ) {
    const hz = new Float32Array(nBands + 2);
    for (let i = 0; i < hz.length; i++) {
      hz[i] = fmin + (fmax - fmin) * (i / (nBands + 1));
    }
    const bins = new Int32Array(hz.length);
    for (let i = 0; i < hz.length; i++)
      bins[i] = Math.floor(((nfft + 1) * hz[i]) / sr);

    const fb = new Array(nBands);
    for (let b = 1; b <= nBands; b++) {
      const f = new Float32Array(nfft >> 1);
      for (let k = bins[b - 1]; k < bins[b]; k++) {
        if (k < f.length)
          f[k] =
            (k - bins[b - 1]) /
            Math.max(1, bins[b] - bins[b - 1]);
      }
      for (let k = bins[b]; k < bins[b + 1]; k++) {
        if (k < f.length)
          f[k] =
            (bins[b + 1] - k) /
            Math.max(1, bins[b + 1] - bins[b]);
      }
      fb[b - 1] = f;
    }
    return fb;
  };

  /* ---------- Apply filterbank ---------- */
  DSP.applyFB = function (mag, fb) {
    const out = new Float32Array(fb.length);
    for (let m = 0; m < fb.length; m++) {
      const w = fb[m];
      let s = 0;
      for (let k = 0; k < w.length && k < mag.length; k++)
        s += w[k] * mag[k];
      out[m] = Math.max(1e-12, s);
    }
    return out;
  };

  /* ---------- DCT-II (naive) ---------- */
  DSP.dct = function (x, kCount) {
    const N = x.length,
      K = Math.min(kCount, N);
    const out = new Float32Array(K);
    const factor = Math.PI / N;
    for (let k = 0; k < K; k++) {
      let s = 0;
      for (let n = 0; n < N; n++)
        s += x[n] * Math.cos((n + 0.5) * k * factor);
      out[k] = s; // không chuẩn hóa; đủ dùng nhất quán tại client
    }
    return out;
  };

  /* ---------- MFCC / LFCC ---------- */
  DSP.mfccFromMag = function (mag, nfft, sr, nMels = 64, nCeps = 13) {
    const fb = DSP.melFilterbank(nfft, sr, nMels);
    const mel = DSP.applyFB(mag, fb);
    for (let i = 0; i < mel.length; i++) mel[i] = Math.log(mel[i]);
    const ceps = DSP.dct(mel, nCeps);
    return ceps;
  };

  DSP.lfccFromMag = function (mag, nfft, sr, nBands = 40, nCeps = 20) {
    const fb = DSP.linearFilterbank(nfft, sr, nBands);
    const lin = DSP.applyFB(mag, fb);
    for (let i = 0; i < lin.length; i++) lin[i] = Math.log(lin[i]);
    const ceps = DSP.dct(lin, nCeps);
    return ceps;
  };

  /* ---------- PCEN (Per-Channel Energy Normalization) ---------- */
  DSP.createPCENState = function (
    nBands,
    alpha = 0.98,
    delta = 2.0,
    r = 0.5,
    eps = 1e-6,
    emaBeta = 0.1
  ) {
    return {
      alpha,
      delta,
      r,
      eps,
      emaBeta,
      ema: new Float32Array(nBands).fill(0),
    };
  };

  DSP.pcenApply = function (bandPow /* Float32Array */, state /* from createPCENState */) {
    const out = new Float32Array(bandPow.length);
    for (let m = 0; m < bandPow.length; m++) {
      const x = bandPow[m];
      state.ema[m] =
        (1 - state.emaBeta) * state.ema[m] + state.emaBeta * x;
      const norm =
        x / Math.pow(state.eps + state.ema[m], state.alpha);
      out[m] =
        Math.pow(norm + state.delta, state.r) -
        Math.pow(state.delta, state.r);
    }
    return out;
  };

  /* ---------- Spectral features ---------- */
  DSP.zcr = function (frame) {
    let c = 0;
    for (let i = 1; i < frame.length; i++) {
      if ((frame[i - 1] >= 0) !== (frame[i] >= 0)) c++;
    }
    return c / frame.length;
  };

  DSP.spectralFlatness = function (mag) {
    const eps = 1e-12;
    let geo = 0,
      arith = 0;
    for (let i = 0; i < mag.length; i++) {
      const p = mag[i] * mag[i] + eps;
      geo += Math.log(p);
      arith += p;
    }
    geo = Math.exp(geo / mag.length);
    arith = arith / mag.length + eps;
    return geo / arith;
  };

  DSP.spectralRolloff = function (mag, roll = 0.85) {
    const N = mag.length;
    let total = 0;
    for (let i = 0; i < N; i++) total += mag[i];
    let thr = total * roll,
      acc = 0;
    for (let i = 0; i < N; i++) {
      acc += mag[i];
      if (acc >= thr) return i / N;
    }
    return 1.0;
  };

  DSP.spectralEntropy = function (mag, nBlocks = 10) {
    const N = mag.length,
      eps = 1e-12;
    let sum = 0;
    for (let i = 0; i < N; i++) sum += mag[i];
    if (sum <= 0) return 0;
    const block = Math.floor(N / nBlocks) || 1;
    let H = 0;
    for (let b = 0; b < nBlocks; b++) {
      let s = 0;
      const st = b * block,
        en = b === nBlocks - 1 ? N : st + block;
      for (let k = st; k < en; k++) s += mag[k];
      const p = s / sum + eps;
      H += -p * Math.log2(p);
    }
    return H / Math.log2(nBlocks); // normalize to [0,1]
  };

  DSP.spectralContrast = function (mag, nBands = 6) {
    // Simple approx: split into bands, take (max-min)/mean per band then average
    const N = mag.length;
    const bandSize = Math.floor(N / nBands) || 1;
    let acc = 0;
    for (let b = 0; b < nBands; b++) {
      const st = b * bandSize,
        en = b === nBands - 1 ? N : st + bandSize;
      let minv = 1e9,
        maxv = -1e9,
        mean = 0,
        cnt = en - st;
      for (let k = st; k < en; k++) {
        const v = mag[k];
        if (v < minv) minv = v;
        if (v > maxv) maxv = v;
        mean += v;
      }
      mean = mean / Math.max(1, cnt);
      const c = mean > 0 ? (maxv - minv) / mean : 0;
      acc += c;
    }
    return acc / nBands;
  };

  /* ---------- F0 (autocorrelation/YIN-ish simplified) ---------- */
  DSP.f0Autocorr = function (frame, sr, fmin = 60, fmax = 400) {
    // Pre-emphasis (light)
    const pre = new Float32Array(frame.length);
    pre[0] = frame[0];
    for (let i = 1; i < frame.length; i++)
      pre[i] = frame[i] - 0.97 * frame[i - 1];

    const N = pre.length;
    const tauMin = Math.max(1, Math.floor(sr / fmax));
    const tauMax = Math.min(N - 1, Math.floor(sr / fmin));
    let bestTau = -1,
      bestVal = -1;

    for (let tau = tauMin; tau <= tauMax; tau++) {
      let corr = 0;
      for (let i = 0; i < N - tau; i++)
        corr += pre[i] * pre[i + tau];
      if (corr > bestVal) {
        bestVal = corr;
        bestTau = tau;
      }
    }
    if (bestTau <= 0) return 0;
    const f0 = sr / bestTau;
    return f0 >= fmin && f0 <= fmax ? f0 : 0;
  };

  /* ---------- Convenience: MFCC/LFCC from time-domain frame ---------- */
  DSP.mfccFromFrame = function (
    frame,
    sr,
    nfft = 1024,
    nMels = 64,
    nCeps = 13,
    win /* opt */
  ) {
    // zero-pad to nfft
    const buf = new Float32Array(nfft);
    const L = Math.min(frame.length, nfft);
    if (win) {
      for (let i = 0; i < L; i++) buf[i] = frame[i] * win[i];
    } else {
      for (let i = 0; i < L; i++) buf[i] = frame[i];
    }
    const re = buf,
      im = new Float32Array(nfft);
    DSP.fftRadix2(re, im);
    const mag = new Float32Array(nfft >> 1);
    for (let k = 0; k < mag.length; k++)
      mag[k] = Math.hypot(re[k], im[k]);
    return DSP.mfccFromMag(mag, nfft, sr, nMels, nCeps);
    // (ghi chú: dùng trực tiếp magSpectrum + mfccFromMag nếu đã có mag)
  };

  DSP.lfccFromFrame = function (
    frame,
    sr,
    nfft = 1024,
    nBands = 40,
    nCeps = 20,
    win /* opt */
  ) {
    const buf = new Float32Array(nfft);
    const L = Math.min(frame.length, nfft);
    if (win) {
      for (let i = 0; i < L; i++) buf[i] = frame[i] * win[i];
    } else {
      for (let i = 0; i < L; i++) buf[i] = frame[i];
    }
    const re = buf,
      im = new Float32Array(nfft);
    DSP.fftRadix2(re, im);
    const mag = new Float32Array(nfft >> 1);
    for (let k = 0; k < mag.length; k++)
      mag[k] = Math.hypot(re[k], im[k]);
    return DSP.lfccFromMag(mag, nfft, sr, nBands, nCeps);
  };

  // attach to global
  g.DSP = DSP;
})(
  typeof self !== "undefined"
    ? self
    : typeof window !== "undefined"
    ? window
    : this
);
