"""Download and prepare the official CWRU 12 kHz drive-end benchmark.

The raw MATLAB files are downloaded from the Case Western Reserve University
Bearing Data Center. They are not redistributed in this repository.
"""

from __future__ import annotations

import json
import os
import tempfile
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from vibeguard.constants import FEATURE_NAMES
from vibeguard.features import extract_features

SAMPLE_RATE_HZ = 12_000
WINDOW_SIZE = 4_096
WINDOW_STEP = 2_048
CLASS_NAMES = ("healthy", "inner_race", "ball", "outer_race")
BASE_URL = "https://engineering.case.edu/sites/default/files/{file_number}.mat"


@dataclass(frozen=True)
class Recording:
    file_number: int
    class_name: str
    load_hp: int
    rpm: int


# CWRU 12 kHz drive-end recordings: baseline + 0.007-inch single-point faults.
# Each fault class uses the 6 o'clock outer-race position for consistency.
LOAD_RPMS = (1797, 1772, 1750, 1730)
RECORDINGS: tuple[Recording, ...] = tuple(
    [
        Recording(start + load, class_name, load, rpm)
        for start, class_name in (
            (97, "healthy"),
            (105, "inner_race"),
            (118, "ball"),
            (130, "outer_race"),
        )
        for load, rpm in enumerate(LOAD_RPMS)
    ]
)


def split_for_load(load_hp: int) -> str:
    """Use disjoint motor loads for train, validation, and final test."""
    if load_hp in (0, 1):
        return "train"
    if load_hp == 2:
        return "validation"
    if load_hp == 3:
        return "test"
    raise ValueError(f"Unsupported motor load {load_hp}; expected 0, 1, 2, or 3 HP.")


def download_recording(recording: Recording, raw_dir: str | Path, *, timeout: int = 45) -> Path:
    """Download one official MATLAB recording atomically, without overwriting valid data."""
    destination_dir = Path(raw_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{recording.file_number}.mat"
    if destination.is_file() and destination.stat().st_size > 1_000:
        return destination

    url = BASE_URL.format(file_number=recording.file_number)
    request = urllib.request.Request(url, headers={"User-Agent": "VibeGuard-Research-Project/1.0"})
    temp_name: str | None = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "").lower()
            first = response.read(128)
            # MATLAB v5 files begin with a descriptive 116-byte header; HTML error
            # pages must not be saved under a .mat extension.
            if b"html" in first.lower() or "text/html" in content_type:
                raise RuntimeError(
                    "The data host returned HTML instead of file "
                    f"{recording.file_number}."
                )
            file_descriptor, temp_name = tempfile.mkstemp(
                prefix=f".{recording.file_number}.", suffix=".download", dir=destination_dir
            )
            with os.fdopen(file_descriptor, "wb") as output:
                output.write(first)
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        if temp_name is None or Path(temp_name).stat().st_size <= 1_000:
            raise RuntimeError(f"Downloaded file {recording.file_number} is unexpectedly small.")
        os.replace(temp_name, destination)
        temp_name = None
    except Exception:
        if temp_name:
            Path(temp_name).unlink(missing_ok=True)
        raise
    return destination


def _load_drive_end_signal(mat_path: str | Path) -> np.ndarray:
    contents = loadmat(mat_path)
    candidates = [key for key in contents if key.endswith("_DE_time")]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one drive-end time-series key ending in '_DE_time' in {mat_path}; "
            f"found {candidates}."
        )
    signal = np.asarray(contents[candidates[0]], dtype=np.float64).reshape(-1)
    if signal.size < WINDOW_SIZE or not np.isfinite(signal).all():
        raise ValueError(f"Invalid or too-short drive-end signal in {mat_path}.")
    return signal


def windows_from_signal(signal: np.ndarray) -> np.ndarray:
    """Create contiguous windows; every window from a file stays in that file's split."""
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    if values.size < WINDOW_SIZE:
        raise ValueError(f"Signal must contain at least {WINDOW_SIZE} samples.")
    if not np.isfinite(values).all():
        raise ValueError("Signal contains NaN or infinity.")
    windows = np.lib.stride_tricks.sliding_window_view(values, WINDOW_SIZE)[::WINDOW_STEP]
    return np.ascontiguousarray(windows, dtype=np.float64)


def prepare_dataset(
    raw_dir: str | Path,
    output_dir: str | Path,
    *,
    download: bool = True,
) -> dict[str, Any]:
    """Extract features and save a provenance-rich NPZ without mixing recording splits."""
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    feature_rows: list[list[float]] = []
    labels: list[int] = []
    splits: list[str] = []
    recording_ids: list[str] = []
    load_values: list[int] = []
    window_indices: list[int] = []
    recording_summary: list[dict[str, Any]] = []

    for recording in RECORDINGS:
        mat_file = raw_path / f"{recording.file_number}.mat"
        if download:
            mat_file = download_recording(recording, raw_path)
        if not mat_file.is_file():
            raise FileNotFoundError(
                f"Missing {mat_file}. Download the official files or rerun with download=True."
            )
        signal = _load_drive_end_signal(mat_file)
        windows = windows_from_signal(signal)
        label_index = CLASS_NAMES.index(recording.class_name)
        split_name = split_for_load(recording.load_hp)
        fundamental_hz = recording.rpm / 60.0
        for window_index, window in enumerate(windows):
            features = extract_features(
                window,
                fundamental_hz=fundamental_hz,
                sample_rate_hz=SAMPLE_RATE_HZ,
            )
            feature_rows.append([features[name] for name in FEATURE_NAMES])
            labels.append(label_index)
            splits.append(split_name)
            recording_ids.append(str(recording.file_number))
            load_values.append(recording.load_hp)
            window_indices.append(window_index)
        recording_summary.append(
            {
                **asdict(recording),
                "split": split_name,
                "raw_file": mat_file.name,
                "raw_samples": int(signal.size),
                "window_count": int(windows.shape[0]),
            }
        )

    features_array = np.asarray(feature_rows, dtype=np.float32)
    labels_array = np.asarray(labels, dtype=np.int64)
    splits_array = np.asarray(splits, dtype="U10")
    recording_array = np.asarray(recording_ids, dtype="U8")
    loads_array = np.asarray(load_values, dtype=np.int64)
    indices_array = np.asarray(window_indices, dtype=np.int64)
    if not np.isfinite(features_array).all():
        raise FloatingPointError("Feature extraction produced non-finite values.")

    dataset_file = output_path / "cwru_features.npz"
    temporary = dataset_file.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            features=features_array,
            labels=labels_array,
            splits=splits_array,
            recording_ids=recording_array,
            load_hp=loads_array,
            window_indices=indices_array,
        )
    os.replace(temporary, dataset_file)

    summary: dict[str, Any] = {
        "dataset_name": "Case Western Reserve University Bearing Data Center - 12 kHz Drive End",
        "source_url": "https://engineering.case.edu/bearingdatacenter/download-data-file",
        "raw_file_base_url": BASE_URL,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "window_size": WINDOW_SIZE,
        "window_step": WINDOW_STEP,
        "fault_severity_in_inches": 0.007,
        "classes": list(CLASS_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "split_protocol": {
            "train": "0 HP and 1 HP recordings",
            "validation": "2 HP recordings",
            "test": "3 HP recordings",
            "leakage_control": "all windows from a source recording remain in one motor-load split",
        },
        "samples": int(labels_array.size),
        "split_counts": {
            name: int(np.sum(splits_array == name))
            for name in ("train", "validation", "test")
        },
        "class_counts": {
            class_name: int(np.sum(labels_array == index))
            for index, class_name in enumerate(CLASS_NAMES)
        },
        "recordings": recording_summary,
        "data_characterization": (
            "Measured laboratory vibration signals from a motor test rig with seeded bearing "
            "faults. This is not field-factory data and does not establish generalization to "
            "different machines, sensors, mounting conditions, or naturally developing faults."
        ),
        "raw_files_redistributed": False,
    }
    report_file = output_path / "dataset_manifest.json"
    report_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["prepared_dataset"] = str(dataset_file)
    summary["manifest"] = str(report_file)
    return summary


def load_prepared_dataset(path: str | Path) -> dict[str, np.ndarray]:
    """Load and validate a prepared dataset without enabling pickle loading."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as archive:
        required = {"features", "labels", "splits", "recording_ids", "load_hp", "window_indices"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Prepared dataset is missing fields: {sorted(missing)}")
        result = {key: archive[key] for key in required}
    n = result["labels"].size
    if result["features"].shape != (n, len(FEATURE_NAMES)):
        raise ValueError("Feature matrix shape does not match the versioned feature contract.")
    if any(result[key].shape != (n,) for key in required - {"features"}):
        raise ValueError("Prepared dataset arrays have inconsistent row counts.")
    if not np.isfinite(result["features"]).all():
        raise ValueError("Prepared dataset contains non-finite features.")
    if not set(np.unique(result["labels"])).issubset(set(range(len(CLASS_NAMES)))):
        raise ValueError("Prepared dataset contains an unknown class index.")
    return result
