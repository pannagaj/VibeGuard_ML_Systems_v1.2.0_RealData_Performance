"""Command-line entry point for the VibeGuard ML engineering workflow."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from vibeguard.artifacts import load_model_artifact, write_json
from vibeguard.benchmark import benchmark_inference, save_benchmark_report
from vibeguard.constants import CLASS_NAMES, FEATURE_NAMES, SAMPLE_RATE_HZ
from vibeguard.data import generate_dataset, load_dataset, split_dataset
from vibeguard.features import extract_features
from vibeguard.inference import Predictor
from vibeguard.metrics import classification_metrics
from vibeguard.optimization import run_optimization_experiment
from vibeguard.training import train_project


def _add_common_data_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", default="data/motor_vibration.csv", help="Versioned feature CSV.")


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"JSON file not found: {source}")
    with source.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {source}.")
    return payload


def _cmd_generate(args: argparse.Namespace) -> int:
    path = generate_dataset(args.output, args.samples_per_class, args.seed)
    data = load_dataset(path)
    print(
        json.dumps(
            {
                "dataset": str(path),
                "samples": int(data.labels.size),
                "features": len(FEATURE_NAMES),
                "classes": {
                    name: int(np.sum(data.labels == i))
                    for i, name in enumerate(CLASS_NAMES)
                },
                "seed": args.seed,
                "synthetic": True,
            },
            indent=2,
        )
    )
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    config = _read_json(args.config) if args.config else {}
    training_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    baseline_cfg = config.get("baseline", {})

    def selected(value: Any, key: str, section: dict[str, Any], fallback: Any) -> Any:
        return value if value is not None else section.get(key, fallback)

    result = train_project(
        args.data,
        args.output_dir,
        epochs=selected(args.epochs, "epochs", training_cfg, 40),
        batch_size=selected(args.batch_size, "batch_size", training_cfg, 128),
        patience=selected(args.patience, "early_stopping_patience", model_cfg, 8),
        learning_rate=selected(args.learning_rate, "learning_rate", model_cfg, 1e-3),
        seed=selected(args.seed, "seed", training_cfg, 42),
        device_name=selected(args.device, "device", training_cfg, "auto"),
        with_baseline=not args.no_baseline,
        hidden_dims=tuple(model_cfg.get("hidden_dims", [96, 48])),
        dropout=model_cfg.get("dropout", 0.12),
        weight_decay=model_cfg.get("weight_decay", 1e-4),
        gradient_clip_norm=model_cfg.get("gradient_clip_norm", 2.0),
        baseline_estimators=baseline_cfg.get("n_estimators", 250),
    )
    print(json.dumps(result["summary"], indent=2))
    print(f"\nFull metrics: {Path(args.output_dir) / 'model_comparison.json'}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.data)
    predictor = Predictor(args.checkpoint, device=args.device)
    selected_seed = args.seed if args.seed is not None else int(predictor.metadata["seed"])
    test = split_dataset(dataset, seed=selected_seed).test
    metrics = classification_metrics(test.labels, predictor.predict_proba(test.features))
    metrics.update(
        {
            "model": "VibrationMLP",
            "split": "held-out test",
            "test_samples": int(test.labels.size),
            "split_seed": selected_seed,
        }
    )
    write_json(args.output, metrics)
    print(json.dumps(metrics, indent=2))
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    source = Path(args.input)
    if not source.is_file():
        raise FileNotFoundError(f"Input CSV not found: {source}")
    with source.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = set(FEATURE_NAMES) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Input CSV is missing features: {sorted(missing)}")
        source_rows = list(reader)
    if not source_rows:
        raise ValueError("Input CSV contains no records.")
    try:
        matrix = np.asarray(
            [[float(row[name]) for name in FEATURE_NAMES] for row in source_rows], dtype=np.float32
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Input CSV contains a missing or non-numeric feature.") from exc
    if not np.isfinite(matrix).all():
        raise ValueError("Input CSV contains a NaN or infinite feature.")
    predictions = Predictor(args.checkpoint, device=args.device).predict(matrix)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "predicted_label",
        "confidence",
        *[f"probability_{name}" for name in CLASS_NAMES],
    ]
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for input_row, prediction in zip(source_rows, predictions, strict=True):
            writer.writerow(
                {
                    "sample_id": input_row.get("sample_id", ""),
                    "predicted_label": prediction["predicted_label"],
                    "confidence": prediction["confidence"],
                    **{
                        f"probability_{name}": probability
                        for name, probability in prediction["probabilities"].items()
                    },
                }
            )
    print(json.dumps({"predictions": len(predictions), "output": str(output)}, indent=2))
    return 0


def _cmd_predict_window(args: argparse.Namespace) -> int:
    payload = _read_json(args.input)
    signal = np.asarray(payload.get("signal", []), dtype=np.float64)
    features = extract_features(
        signal,
        fundamental_hz=float(payload.get("fundamental_hz", 0.0)),
        sample_rate_hz=int(payload.get("sample_rate_hz", SAMPLE_RATE_HZ)),
    )
    feature_row = np.asarray([[features[name] for name in FEATURE_NAMES]], dtype=np.float32)
    prediction = Predictor(args.checkpoint, device=args.device).predict(feature_row)[0]
    result = {"features": features, "prediction": prediction}
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    report = benchmark_inference(
        args.checkpoint,
        runs=args.runs,
        warmup=args.warmup,
        seed=args.seed,
        thread_count=args.threads,
    )
    save_benchmark_report(report, args.output)
    print(json.dumps(report, indent=2))
    return 0


def _cmd_optimize(args: argparse.Namespace) -> int:
    report = run_optimization_experiment(
        args.checkpoint,
        args.output_dir,
        runs=args.runs,
        warmup=args.warmup,
        seed=args.seed,
        thread_count=args.threads,
        data_path=args.data,
    )
    print(json.dumps(report, indent=2))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.data)
    model, metadata = load_model_artifact(args.checkpoint, device="cpu")
    selected_seed = args.seed if args.seed is not None else int(metadata["seed"])
    splits = split_dataset(dataset, seed=selected_seed)
    result = {
        "status": "valid",
        "dataset_rows": int(dataset.labels.size),
        "feature_count": int(dataset.features.shape[1]),
        "class_counts": {
            name: int(np.sum(dataset.labels == i))
            for i, name in enumerate(CLASS_NAMES)
        },
        "split_sizes": {
            "train": int(splits.train.labels.size),
            "validation": int(splits.validation.labels.size),
            "test": int(splits.test.labels.size),
        },
        "checkpoint_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "checkpoint_schema": metadata["schema_version"],
        "split_seed": selected_seed,
        "model_hidden_dims": metadata["hidden_dims"],
    }
    print(json.dumps(result, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vibeguard",
        description=(
            "Reproducible motor-vibration ML training, evaluation, inference and optimization."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser(
        "generate-data", help="Create reproducible synthetic data."
    )
    generate_parser.add_argument("--output", default="data/motor_vibration.csv")
    generate_parser.add_argument("--samples-per-class", type=int, default=1_000)
    generate_parser.add_argument("--seed", type=int, default=42)
    generate_parser.set_defaults(handler=_cmd_generate)

    train_parser = subparsers.add_parser("train", help="Train the PyTorch model and tree baseline.")
    _add_common_data_argument(train_parser)
    train_parser.add_argument(
        "--config", default="configs/default.json", help="JSON hyperparameter config."
    )
    train_parser.add_argument("--output-dir", default="artifacts")
    train_parser.add_argument("--epochs", type=int, default=None)
    train_parser.add_argument("--batch-size", type=int, default=None)
    train_parser.add_argument("--patience", type=int, default=None)
    train_parser.add_argument("--learning-rate", type=float, default=None)
    train_parser.add_argument("--seed", type=int, default=None)
    train_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    train_parser.add_argument(
        "--no-baseline", action="store_true", help="Skip ExtraTrees comparison."
    )
    train_parser.set_defaults(handler=_cmd_train)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evaluate on the deterministic held-out test split."
    )
    _add_common_data_argument(evaluate_parser)
    evaluate_parser.add_argument("--checkpoint", default="artifacts/model.pt")
    evaluate_parser.add_argument("--output", default="artifacts/evaluation.json")
    evaluate_parser.add_argument("--seed", type=int, default=None)
    evaluate_parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    evaluate_parser.set_defaults(handler=_cmd_evaluate)

    predict_parser = subparsers.add_parser(
        "predict", help="Predict labels for a CSV of feature rows."
    )
    predict_parser.add_argument("--input", required=True)
    predict_parser.add_argument("--checkpoint", default="artifacts/model.pt")
    predict_parser.add_argument("--output", default="artifacts/predictions.csv")
    predict_parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    predict_parser.set_defaults(handler=_cmd_predict)

    window_parser = subparsers.add_parser(
        "predict-window", help="Extract features and classify one raw signal window."
    )
    window_parser.add_argument(
        "--input",
        required=True,
        help="JSON with signal, fundamental_hz and optional sample_rate_hz.",
    )
    window_parser.add_argument("--checkpoint", default="artifacts/model.pt")
    window_parser.add_argument("--output", default=None, help="Optional JSON output path.")
    window_parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    window_parser.set_defaults(handler=_cmd_predict_window)

    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Measure eager CPU batch latency and throughput."
    )
    benchmark_parser.add_argument("--checkpoint", default="artifacts/model.pt")
    benchmark_parser.add_argument("--output", default="artifacts/benchmark.json")
    benchmark_parser.add_argument("--runs", type=int, default=250)
    benchmark_parser.add_argument("--warmup", type=int, default=40)
    benchmark_parser.add_argument("--threads", type=int, default=1)
    benchmark_parser.add_argument("--seed", type=int, default=42)
    benchmark_parser.set_defaults(handler=_cmd_benchmark)

    optimize_parser = subparsers.add_parser(
        "optimize", help="Export and compare torch.export and dynamic INT8 CPU inference."
    )
    optimize_parser.add_argument("--checkpoint", default="artifacts/model.pt")
    optimize_parser.add_argument(
        "--data", default=None, help="Optional dataset for held-out quality evaluation."
    )
    optimize_parser.add_argument("--output-dir", default="artifacts/optimization")
    optimize_parser.add_argument("--runs", type=int, default=300)
    optimize_parser.add_argument("--warmup", type=int, default=40)
    optimize_parser.add_argument("--threads", type=int, default=1)
    optimize_parser.add_argument("--seed", type=int, default=42)
    optimize_parser.set_defaults(handler=_cmd_optimize)

    validate_parser = subparsers.add_parser(
        "validate", help="Check dataset, split integrity and model artifact."
    )
    _add_common_data_argument(validate_parser)
    validate_parser.add_argument("--checkpoint", default="artifacts/model.pt")
    validate_parser.add_argument("--seed", type=int, default=None)
    validate_parser.set_defaults(handler=_cmd_validate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (
        ValueError,
        FileNotFoundError,
        RuntimeError,
        FloatingPointError,
        KeyError,
        OSError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
