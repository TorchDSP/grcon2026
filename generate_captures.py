"""Generate the SigMF captures for the XCiT modulation CTF challenge.

This script intentionally uses the TorchSig 2.1.1 signal vocabulary that was
used to train ``xcit_narrowband_v1.0.0.ckpt``. GNU Radio writes the complex64
sample streams; SigMF supplies the recording metadata.

Run with GNU Radio's Python environment:
    python generate_captures.py --output-dir captures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchsig
from torchsig.datasets.datasets import TorchSigIterableDataset
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.utils.defaults import TorchSigDefaults


SAMPLE_RATE = 10_000_000
NUM_SAMPLES = 4096
CAPTURES = (
    ("capture_01", "fm", 3001),
    ("capture_02", "lfm-data", 3002),
    ("capture_03", "ook", 3003),
    ("capture_04", "am-dsb", 3004),
    ("capture_05", "tone", 3006),
)


def parse_args() -> argparse.Namespace:
    """Parse capture-generation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("captures"))
    parser.add_argument(
        "--writer",
        choices=["gnuradio", "numpy"],
        default="gnuradio",
        help="Use numpy only for development environments without GNU Radio.",
    )
    return parser.parse_args()


def dataset_metadata() -> dict[str, object]:
    """Return the narrowband configuration used for every capture."""
    return {
        **TorchSigDefaults().default_dataset_metadata,
        "sample_rate": SAMPLE_RATE,
        "num_iq_samples_dataset": NUM_SAMPLES,
        "num_signals_min": 1,
        "num_signals_max": 1,
        "cochannel_overlap_probability": 0.0,
        "snr_db_min": 30.0,
        "snr_db_max": 50.0,
        "noise_power_db": 0.0,
        "signal_duration_in_samples_min": NUM_SAMPLES,
        "signal_duration_in_samples_max": NUM_SAMPLES,
        "bandwidth_min": 2_500_000,
        "bandwidth_max": 3_333_333,
        "signal_center_freq_min": -250_000,
        "signal_center_freq_max": 250_000,
        "frequency_min": -5_000_000,
        "frequency_max": 4_999_999,
    }


def generate_iq(signal_generator: str, seed: int) -> np.ndarray:
    """Generate one seeded TorchSig narrowband example."""
    dataset = TorchSigIterableDataset(
        metadata=dataset_metadata(),
        signal_generators=[signal_generator],
        target_labels=["class_index"],
    )
    loader = WorkerSeedingDataLoader(
        dataset,
        batch_size=1,
        collate_fn=lambda batch: batch,
        num_workers=0,
        seed=seed,
    )
    loader.seed(seed)
    iq, _ = next(iter(loader))[0]
    return np.asarray(iq, dtype=np.complex64)


def write_with_gnuradio(iq: np.ndarray, path: Path) -> None:
    """Write complex samples through a minimal GNU Radio flowgraph."""
    try:
        from gnuradio import blocks, gr
    except ImportError as error:
        raise RuntimeError(
            "GNU Radio is unavailable. Run in a GNU Radio environment or use "
            "--writer numpy for local development."
        ) from error

    flowgraph = gr.top_block()
    source = blocks.vector_source_c(iq.tolist(), repeat=False)
    sink = blocks.file_sink(gr.sizeof_gr_complex, str(path), False)
    flowgraph.connect(source, sink)
    flowgraph.run()


def write_sigmf(base_path: Path, iq: np.ndarray, writer: str) -> None:
    """Write one complex64 data file and its SigMF metadata."""
    data_path = base_path.with_suffix(".sigmf-data")
    meta_path = base_path.with_suffix(".sigmf-meta")
    if writer == "gnuradio":
        write_with_gnuradio(iq, data_path)
    else:
        iq.astype("<c8").tofile(data_path)

    metadata = {
        "global": {
            "core:datatype": "cf32_le",
            "core:sample_rate": SAMPLE_RATE,
            "core:version": "1.2.5",
            "core:description": "GNU Radio Conference 2026 CTF narrowband capture",
        },
        "captures": [{"core:sample_start": 0, "core:frequency": 0.0}],
        "annotations": [],
    }
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def write_metadata_index(output_dir: Path) -> None:
    """Write the unlabeled dataset index required by TorchSig's SigMFReader."""
    rows = [
        f"{index},unknown,-1,{SAMPLE_RATE}\n"
        for index, _capture in enumerate(CAPTURES)
    ]
    (output_dir / "metadata.csv").write_text("".join(rows), encoding="utf-8")


def main() -> None:
    """Generate all challenge captures in their decoding order."""
    args = parse_args()
    if not torchsig.__version__.startswith("2.1."):
        raise RuntimeError(
            "Generate these captures with TorchSig 2.1.x to match the released "
            f"checkpoint; found {torchsig.__version__}."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(2026)
    for filename, signal_generator, seed in CAPTURES:
        write_sigmf(
            args.output_dir / filename,
            generate_iq(signal_generator, seed),
            args.writer,
        )
        print(f"wrote {filename}.sigmf-data and {filename}.sigmf-meta")
    write_metadata_index(args.output_dir)
    print("wrote metadata.csv")


if __name__ == "__main__":
    main()
