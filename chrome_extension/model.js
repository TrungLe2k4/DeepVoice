// model.js (placeholder). For real use: load ONNX model via onnxruntime-web.
export async function predictProb(features) {
  // Simple logistic-ish score with clamped output, purely demo.
  // Replace with real inference call.
  const w = {
    rms: -3.5,    // softer speech more suspicious
    zcr: 8.0,     // higher zero-crossing rate -> more suspicious (buzziness)
    flat: 2.5,    // spectral flatness high -> suspicious
    kurt: -0.8,   // lower kurtosis -> suspicious
    mean: 0.0
  };
  const b = -1.2;
  const z = (features.rms||0)*w.rms + (features.zcr||0)*w.zcr + (features.flat||0)*w.flat + (features.kurt||0)*w.kurt + b;
  const prob = 1.0 / (1.0 + Math.exp(-z));
  // clamp & sanity
  return Math.max(0, Math.min(1, prob));
}
