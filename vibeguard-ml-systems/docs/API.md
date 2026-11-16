# Inference API

Run the service from the project root after training or using the bundled reference artifact:

```bash
VIBEGUARD_CHECKPOINT=artifacts/model.pt uvicorn vibeguard.service:app --host 127.0.0.1 --port 8000
```

Interactive schema: `http://127.0.0.1:8000/docs`.

## `GET /health`

Returns `ready`, `model_missing`, or `model_invalid`. A ready response means the configured checkpoint exists and can be loaded.

## `GET /metadata`

Returns the artifact's project/schema version, ordered feature names, class names and CPU inference target.

## `POST /predict`

Accepts 1–256 feature dictionaries. Each record must include exactly the ordered 20 feature names (JSON object key order is not required; server order is canonical). Unknown or missing features are rejected.

```json
{
  "samples": [
    {
      "rms": 0.2,
      "std": 0.2,
      "mean_absolute": 0.1,
      "peak_to_peak": 0.8,
      "peak_absolute": 0.4,
      "skewness": 0.0,
      "kurtosis": 1.8,
      "crest_factor": 2.0,
      "zero_crossing_rate": 0.1,
      "spectral_centroid_hz": 40.0,
      "spectral_entropy": 0.3,
      "dominant_frequency_hz": 20.0,
      "spectral_rolloff_85_hz": 60.0,
      "band_power_0_60": 0.8,
      "band_power_60_140": 0.1,
      "band_power_140_260": 0.08,
      "band_power_260_nyquist": 0.02,
      "fundamental_ratio": 0.7,
      "second_harmonic_ratio": 0.1,
      "third_harmonic_ratio": 0.03
    }
  ]
}
```

The numeric values above are illustrative request-shape values, not a labelled sample or a model-performance claim. Use features extracted by VibeGuard or the included example workflow for meaningful predictions.

## `POST /predict-window`

Accepts `signal` (at least 32 finite floats), `fundamental_hz` (> 0), and optional `sample_rate_hz` (> 0, default 1024). It extracts the fixed feature vector then returns both that feature dictionary and the prediction.

```bash
curl -X POST http://127.0.0.1:8000/predict-window \
  -H "Content-Type: application/json" \
  --data-binary @examples/sample_vibration_window.json
```

## Security and operational notes

The included app is designed for local evaluation. It does not implement authentication, authorization, rate limiting, TLS termination, request logging, model hot swapping or multi-tenant isolation. Add those controls before an internet-facing deployment.
