// worklet-processor.js
class DVAnalyzerProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.sampleRate = options?.processorOptions?.sampleRate || 48000;
    this.buf = new Float32Array(this.sampleRate); // ~1s buffer mono
    this.ptr = 0;
    this.lastSent = 0;
  }
  static get parameterDescriptors() { return []; }
  process(inputs, outputs, parameters) {
    const input = inputs[0];
    // mixdown to mono
    const ch0 = input && input[0] ? input[0] : null;
    if (!ch0) return true;
    const out = outputs[0];
    // passthrough
    for (let c = 0; c < out.length; c++) {
      out[c].set(input[c] || ch0);
    }
    // copy to ring buffer
    const frame = ch0;
    const N = frame.length;
    for (let i = 0; i < N; i++) {
      this.buf[this.ptr++] = frame[i];
      if (this.ptr >= this.buf.length) {
        // compute features for last 1 second
        const feats = this.computeFeatures(this.buf);
        this.port.postMessage({ type: "features", features: feats, explain: feats._explain || "" });
        this.ptr = 0;
      }
    }
    return true;
  }

  computeFeatures(x) {
    // Simple features: RMS, ZCR, Spectral Flatness approx, Kurtosis
    const n = x.length;
    let sum = 0, sumsq = 0, zc = 0;
    for (let i = 1; i < n; i++) {
      const xi = x[i];
      sum += xi;
      sumsq += xi*xi;
      if ((x[i-1] >= 0 && xi < 0) || (x[i-1] < 0 && xi >= 0)) zc++;
    }
    const mean = sum / n;
    const rms = Math.sqrt(sumsq / n);
    // spectral flatness approx via log energy ratios (poor man's flatness)
    let absSum = 0, absMax = 1e-9;
    for (let i = 0; i < n; i++) {
      const v = Math.abs(x[i]);
      absSum += v;
      if (v > absMax) absMax = v;
    }
    const flat = (absSum / n) / absMax; // [0..1], flatter -> closer to small value
    // kurtosis
    let m4 = 0, varsum = 0;
    for (let i = 0; i < n; i++) {
      const d = x[i] - mean;
      varsum += d*d;
      m4 += d*d*d*d;
    }
    const variance = varsum / (n-1);
    const kurt = variance > 0 ? (m4 / n) / (variance*variance) : 0;
    const zcr = zc / n;

    const feat = { rms, zcr, flat, kurt, mean };
    // crude explain string
    let ex = "";
    if (flat > 0.5 && zcr > 0.07 && rms < 0.05) ex = "Phẳng phổ cao + ZCR cao + biên độ thấp (đặc trưng tổng hợp?)";
    else if (kurt < 2) ex = "Kurtosis thấp (giọng thiếu đỉnh nhọn tự nhiên)";
    feat._explain = ex;
    return feat;
  }
}
registerProcessor("dv-analyzer", DVAnalyzerProcessor);
