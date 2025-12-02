// worklet-processor.js — DeepVoice Guard (full features, no-echo, 0.5s tick)
/* global registerProcessor, AudioWorkletProcessor, currentTime, sampleRate */
"use strict";

/* ====================== MINI DSP LIB (thay cho importScripts) ====================== */
(function (g) {
  const DSP = {};

  /* ---------- Window ---------- */
  DSP.hanning = function (n) {
    const w = new Float32Array(n);
    const f = (2 * Math.PI) / (n - 1);
    for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos(f * i);
    return w;
  };

  /* ---------- FFT (radix-2, in-place, real->complex arrays) ---------- */
  DSP.fftRadix2 = function (re, im) {
    const n = re.length;
    let i = 0, j = 0;
    // bit-reversal permutation
    for (i = 1; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        let tr = re[i]; re[i] = re[j]; re[j] = tr;
        let ti = im[i]; im[i] = im[j]; im[j] = ti;
      }
    }
    // Cooley–Tukey
    for (let len = 2; len <= n; len <<= 1) {
      const ang = (-2 * Math.PI) / len;
      const wlen_r = Math.cos(ang), wlen_i = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let w_r = 1, w_i = 0;
        for (let j = 0; j < (len >> 1); j++) {
          const u_r = re[i + j], u_i = im[i + j];
          const v_r = re[i + j + (len >> 1)] * w_r - im[i + j + (len >> 1)] * w_i;
          const v_i = re[i + j + (len >> 1)] * w_i + im[i + j + (len >> 1)] * w_r;
          re[i + j] = u_r + v_r;      im[i + j] = u_i + v_i;
          re[i + j + (len >> 1)] = u_r - v_r;  im[i + j + (len >> 1)] = u_i - v_i;
          // w *= wlen
          const nw_r = w_r * wlen_r - w_i * wlen_i;
          const nw_i = w_r * wlen_i + w_i * wlen_r;
          w_r = nw_r; w_i = nw_i;
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
      const r = re[k], ii = im[k];
      out[k] = Math.hypot(r, ii);
    }
    return out;
  };

  /* ---------- Mel helpers ---------- */
  DSP.hzToMel = (hz) => 2595 * Math.log10(1 + hz / 700);
  DSP.melToHz = (mel) => 700 * (Math.pow(10, mel / 2595) - 1);

  /* ---------- Mel Filterbank (triangular) ---------- */
  DSP.melFilterbank = function (nfft, sr, nMels = 64, fmin = 50, fmax = sr / 2) {
    const mMin = DSP.hzToMel(fmin), mMax = DSP.hzToMel(fmax);
    const mPts = new Float32Array(nMels + 2);
    for (let i = 0; i < mPts.length; i++) {
      mPts[i] = mMin + (mMax - mMin) * (i / (nMels + 1));
    }
    const hz = new Float32Array(mPts.length);
    for (let i = 0; i < mPts.length; i++) hz[i] = DSP.melToHz(mPts[i]);

    const bins = new Int32Array(hz.length);
    for (let i = 0; i < hz.length; i++) bins[i] = Math.floor(((nfft + 1) * hz[i]) / sr);

    const fb = new Array(nMels);
    for (let m = 1; m <= nMels; m++) {
      const f = new Float32Array(nfft >> 1);
      for (let k = bins[m - 1]; k < bins[m]; k++) {
        if (k < f.length)
          f[k] = (k - bins[m - 1]) / Math.max(1, bins[m] - bins[m - 1]);
      }
      for (let k = bins[m]; k < bins[m + 1]; k++) {
        if (k < f.length)
          f[k] = (bins[m + 1] - k) / Math.max(1, bins[m + 1] - bins[m]);
      }
      fb[m - 1] = f;
    }
    return fb;
  };

  /* ---------- Linear Filterbank (LFCC) ---------- */
  DSP.linearFilterbank = function (nfft, sr, nBands = 40, fmin = 50, fmax = sr / 2) {
    const hz = new Float32Array(nBands + 2);
    for (let i = 0; i < hz.length; i++) {
      hz[i] = fmin + (fmax - fmin) * (i / (nBands + 1));
    }
    const bins = new Int32Array(hz.length);
    for (let i = 0; i < hz.length; i++) bins[i] = Math.floor(((nfft + 1) * hz[i]) / sr);

    const fb = new Array(nBands);
    for (let b = 1; b <= nBands; b++) {
      const f = new Float32Array(nfft >> 1);
      for (let k = bins[b - 1]; k < bins[b]; k++) {
        if (k < f.length)
          f[k] = (k - bins[b - 1]) / Math.max(1, bins[b] - bins[b - 1]);
      }
      for (let k = bins[b]; k < bins[b + 1]; k++) {
        if (k < f.length)
          f[k] = (bins[b + 1] - k) / Math.max(1, bins[b + 1] - bins[b]);
      }
      fb[b - 1] = f;
    }
    return fb;
  };

  /* ---------- Apply filterbank ---------- */
  DSP.applyFB = function (mag, fb) {
    const out = new Float32Array(fb.length);
    for (let m = 0; m < fb.length; m++) {
      const w = fb[m]; let s = 0;
      for (let k = 0; k < w.length && k < mag.length; k++) s += w[k] * mag[k];
      out[m] = Math.max(1e-12, s);
    }
    return out;
  };

  /* ---------- DCT-II (naive) ---------- */
  DSP.dct = function (x, kCount) {
    const N = x.length, K = Math.min(kCount, N);
    const out = new Float32Array(K);
    const factor = Math.PI / N;
    for (let k = 0; k < K; k++) {
      let s = 0;
      for (let n = 0; n < N; n++) s += x[n] * Math.cos((n + 0.5) * k * factor);
      out[k] = s;
    }
    return out;
  };

  /* ---------- MFCC / LFCC ---------- */
  DSP.mfccFromMag = function (mag, nfft, sr, nMels = 64, nCeps = 13) {
    const fb = DSP.melFilterbank(nfft, sr, nMels);
    const mel = DSP.applyFB(mag, fb);
    for (let i = 0; i < mel.length; i++) mel[i] = Math.log(mel[i]);
    return DSP.dct(mel, nCeps);
  };

  DSP.lfccFromMag = function (mag, nfft, sr, nBands = 40, nCeps = 20) {
    const fb = DSP.linearFilterbank(nfft, sr, nBands);
    const lin = DSP.applyFB(mag, fb);
    for (let i = 0; i < lin.length; i++) lin[i] = Math.log(lin[i]);
    return DSP.dct(lin, nCeps);
  };

  /* ---------- PCEN ---------- */
  DSP.createPCENState = function (nBands, alpha = 0.98, delta = 2.0, r = 0.5, eps = 1e-6, emaBeta = 0.1) {
    return {
      alpha, delta, r, eps, emaBeta,
      ema: new Float32Array(nBands).fill(0)
    };
  };
  DSP.pcenApply = function (bandPow, state) {
    const out = new Float32Array(bandPow.length);
    for (let m = 0; m < bandPow.length; m++) {
      const x = bandPow[m];
      state.ema[m] = (1 - state.emaBeta) * state.ema[m] + state.emaBeta * x;
      const norm = x / Math.pow(state.eps + state.ema[m], state.alpha);
      out[m] = Math.pow(norm + state.delta, state.r) - Math.pow(state.delta, state.r);
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
    let geo = 0, arith = 0;
    for (let i = 0; i < mag.length; i++) {
      const p = mag[i] * mag[i] + eps;
      geo += Math.log(p); arith += p;
    }
    geo = Math.exp(geo / mag.length);
    arith = arith / mag.length + eps;
    return geo / arith;
  };

  DSP.spectralRolloff = function (mag, roll = 0.85) {
    const N = mag.length;
    let total = 0; for (let i = 0; i < N; i++) total += mag[i];
    let thr = total * roll, acc = 0;
    for (let i = 0; i < N; i++) { acc += mag[i]; if (acc >= thr) return i / N; }
    return 1.0;
  };

  DSP.spectralEntropy = function (mag, nBlocks = 10) {
    const N = mag.length, eps = 1e-12;
    let sum = 0; for (let i = 0; i < N; i++) sum += mag[i];
    if (sum <= 0) return 0;
    const block = Math.floor(N / nBlocks) || 1;
    let H = 0;
    for (let b = 0; b < nBlocks; b++) {
      let s = 0;
      const st = b * block, en = (b === nBlocks - 1) ? N : st + block;
      for (let k = st; k < en; k++) s += mag[k];
      const p = s / sum + eps;
      H += -p * Math.log2(p);
    }
    return H / Math.log2(nBlocks);
  };

  DSP.spectralContrast = function (mag, nBands = 6) {
    const N = mag.length;
    const bandSize = Math.floor(N / nBands) || 1;
    let acc = 0;
    for (let b = 0; b < nBands; b++) {
      const st = b * bandSize, en = (b === nBands - 1) ? N : st + bandSize;
      let minv = 1e9, maxv = -1e9, mean = 0, cnt = en - st;
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

  /* ---------- F0 (autocorrelation simplified) ---------- */
  DSP.f0Autocorr = function (frame, sr, fmin = 60, fmax = 400) {
    const pre = new Float32Array(frame.length);
    pre[0] = frame[0];
    for (let i = 1; i < frame.length; i++) pre[i] = frame[i] - 0.97 * frame[i - 1];

    const N = pre.length;
    const tauMin = Math.max(1, Math.floor(sr / fmax));
    const tauMax = Math.min(N - 1, Math.floor(sr / fmin));
    let bestTau = -1, bestVal = -1;

    for (let tau = tauMin; tau <= tauMax; tau++) {
      let corr = 0;
      for (let i = 0; i < N - tau; i++) corr += pre[i] * pre[i + tau];
      if (corr > bestVal) { bestVal = corr; bestTau = tau; }
    }
    if (bestTau <= 0) return 0;
    const f0 = sr / bestTau;
    return (f0 >= fmin && f0 <= fmax) ? f0 : 0;
  };

  g.DSP = DSP;
})(
  typeof globalThis !== "undefined"
    ? globalThis
    : self
);

/* ====================== RING BUFFER ====================== */
class RingBuffer {
  constructor(cap, dim) {
    this.cap = cap; this.dim = dim;
    this.buf = new Float32Array(cap * dim);
    this.size = 0; this.head = 0;
  }
  push(vec) {
    const d = this.dim;
    const idx = (this.head % this.cap) * d;
    this.buf.set(vec, idx);
    this.head = (this.head + 1) % this.cap;
    this.size = Math.min(this.size + 1, this.cap);
  }
  meanStd() {
    const n = this.size, d = this.dim;
    const mean = new Float32Array(d);
    const m2 = new Float32Array(d);
    if (n === 0) return { mean, std: new Float32Array(d) };
    for (let i = 0; i < n; i++) {
      const base = (i * d);
      for (let k = 0; k < d; k++) {
        const x = this.buf[base + k];
        const delta = x - mean[k];
        mean[k] += delta / (i + 1);
        m2[k] += delta * (x - mean[k]);
      }
    }
    const std = new Float32Array(d);
    for (let k = 0; k < d; k++) std[k] = Math.sqrt(Math.max(0, m2[k] / Math.max(1, n - 1)));
    return { mean, std };
  }
}

/* ====================== DVAnalyzer ====================== */
class DVAnalyzer extends AudioWorkletProcessor {
  constructor(options) {
    super();

    const optSr = options?.processorOptions?.sampleRate;
    this.sr = optSr || (typeof sampleRate !== "undefined" ? sampleRate : 48000);

    this.hopSec = 0.5;                 // phân tích khung 0.5s
    this.bufTarget = Math.floor(this.sr * this.hopSec);
    this.frame = new Float32Array(0);

    // FFT/feature params
    this.nfft = 1024;
    this.win = DSP.hanning(this.nfft);
    this.nMels = 64;
    this.melFB = DSP.melFilterbank(this.nfft, this.sr, this.nMels, 50, 8000);
    this.pcenState = DSP.createPCENState(this.nMels, 0.98, 2.0, 0.5, 1e-6, 0.1);

    // Rolling stats (≈ 2s = 4 hop x 0.5s)
    this.pcenHist = new RingBuffer(4, this.nMels);
    this.mfccHist = new RingBuffer(4, 13); // để tính Δ, ΔΔ

    // Smoothing
    this.smooth = { rms: 0, zcr: 0, alpha: 0.2 };

    // noise floor for SNR
    this.noiseEma = 1e-6;
    this.noiseBeta = 0.05;

    this._tmpFrame = null;
    this._lastMfcc = null;
    this._lastDelta = null;
    this._lastEntropy = 0;
    this._lastFlat = 0;
    this._lastF0 = 0;
    this._lastRms = 0;

    // ⏱ Throttle theo thời gian: chỉ postMessage mỗi X giây (worklet-level)
    this.postIntervalSec = 0.5;  // 0.5s → khớp hopSec (mỗi khung 1 lần)
    this.postIntervalFrames = Math.max(
      1,
      Math.round(this.postIntervalSec / this.hopSec)
    ); // hopSec=0.5 → 1 frame
    this._frameIndex = 0;
  }

  _magFromTime(frame) {
    const buf = new Float32Array(this.nfft);
    const L = Math.min(frame.length, this.nfft);
    for (let i = 0; i < L; i++) buf[i] = frame[i] * this.win[i];
    const re = buf, im = new Float32Array(this.nfft);
    DSP.fftRadix2(re, im);
    const mag = new Float32Array(this.nfft >> 1);
    for (let k = 0; k < mag.length; k++) mag[k] = Math.hypot(re[k], im[k]);
    return mag;
  }

  _delta(vecPrev, vecCurr) {
    const d = vecCurr.length;
    const out = new Float32Array(d);
    if (!vecPrev) return out;
    for (let i = 0; i < d; i++) out[i] = vecCurr[i] - vecPrev[i];
    return out;
  }
  _deltaDelta(prevDelta, currDelta) {
    if (!prevDelta) return new Float32Array(currDelta.length);
    const d = currDelta.length, out = new Float32Array(d);
    for (let i = 0; i < d; i++) out[i] = currDelta[i] - prevDelta[i];
    return out;
  }

  _spectralStats(mag) {
    const zcr = DSP.zcr(this._tmpFrame || mag);
    const flat = DSP.spectralFlatness(mag);
    const roll = DSP.spectralRolloff(mag, 0.85);
    const ent  = DSP.spectralEntropy(mag, 10);
    const contr= DSP.spectralContrast(mag, 6);
    return { zcr, flat, rolloff: roll, entropy: ent, contrast: contr };
  }

  _prosody(frame) {
    const f0 = DSP.f0Autocorr(frame, this.sr, 60, 400);

    const N = frame.length;
    let sum = 0; for (let i = 0; i < N; i++) sum += frame[i] * frame[i];
    const rms = Math.sqrt(sum / N);

    const jitter = Math.min(5, Math.abs((f0 - (this._lastF0 || f0)) / Math.max(60, f0 || 60))) * 100;
    const shimmer = Math.min(5, Math.abs((rms - (this._lastRms || rms)) / Math.max(1e-6, rms))) * 100;
    this._lastF0 = f0; this._lastRms = rms;

    const cpp = Math.max(0, 20 - 10 * (this._lastEntropy || 0) - 5 * (this._lastFlat || 0));
    return { f0, jitter, shimmer, cpp, rms };
  }

  _snrEstimate(rms) {
    this.noiseEma = (1 - this.noiseBeta) * this.noiseEma + this.noiseBeta * Math.max(1e-9, rms * rms);
    const snrLin = Math.max(1e-6, (rms * rms) / Math.max(1e-9, this.noiseEma));
    const snrDb = 10 * Math.log10(snrLin);
    return Math.max(0, Math.min(40, snrDb));
  }

  process(inputs) {
    const ch0 = inputs?.[0]?.[0];
    if (!ch0) return true;

    const merged = new Float32Array(this.frame.length + ch0.length);
    merged.set(this.frame, 0); merged.set(ch0, this.frame.length);
    this.frame = merged;

    while (this.frame.length >= this.bufTarget) {
      const slice = this.frame.subarray(0, this.bufTarget);
      this.frame = this.frame.subarray(this.bufTarget);

      this._tmpFrame = slice;

      const mag = this._magFromTime(slice);

      const melPow = DSP.applyFB(mag, this.melFB);
      const pcen = DSP.pcenApply(melPow, this.pcenState);
      this.pcenHist.push(pcen);
      const { mean: pcenMean, std: pcenStd } = this.pcenHist.meanStd();

      const mfcc13 = DSP.mfccFromMag(mag, this.nfft, this.sr, this.nMels, 13);
      let delta13, deltadelta13;
      if (this.mfccHist.size > 0) {
        const prev = this._lastMfcc || null;
        delta13 = this._delta(prev, mfcc13);
        deltadelta13 = this._deltaDelta(this._lastDelta || null, delta13);
      } else {
        delta13 = new Float32Array(13);
        deltadelta13 = new Float32Array(13);
      }
      this.mfccHist.push(mfcc13);
      this._lastMfcc = mfcc13;
      this._lastDelta = delta13;

      const lfcc20 = DSP.lfccFromMag(mag, this.nfft, this.sr, 40, 20);

      const spec = this._spectralStats(mag);
      this._lastEntropy = spec.entropy;
      this._lastFlat = spec.flat;

      const pros = this._prosody(slice);
      const rmsNow = pros.rms;

      this.smooth.rms = this.smooth.alpha * rmsNow + (1 - this.smooth.alpha) * this.smooth.rms;
      this.smooth.zcr = this.smooth.alpha * spec.zcr + (1 - this.smooth.alpha) * this.smooth.zcr;

      const snrDb = this._snrEstimate(rmsNow);

      // ⏱ Throttle theo thời gian ở cấp worklet:
      this._frameIndex++;
      if (this._frameIndex % this.postIntervalFrames !== 0) {
        continue;
      }

      const mfcc39 = new Float32Array(39);
      mfcc39.set(mfcc13, 0);
      mfcc39.set(delta13, 13);
      mfcc39.set(deltadelta13, 26);

      const payload = {
        mfcc: Array.from(mfcc39),
        lfcc: Array.from(lfcc20),
        pcen_stats: { mean: Array.from(pcenMean), std: Array.from(pcenStd) },
        spec: {
          zcr: this.smooth.zcr,
          flat: spec.flat,
          rolloff: spec.rolloff,
          entropy: spec.entropy,
          contrast: spec.contrast
        },
        prosody: {
          f0: pros.f0,
          jitter: pros.jitter,
          shimmer: pros.shimmer,
          cpp: pros.cpp
        },
        meta: { sr: this.sr, win: this.nfft, hop: Math.floor(this.nfft / 4), snr: snrDb }
      };

      this.port.postMessage({ type: "features", payload });
    }

    return true;
  }
}

registerProcessor("dv-analyzer", DVAnalyzer);
