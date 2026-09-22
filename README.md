# TorchSig Workshop: GRCon 2026

Materials for the TorchSig workshop at
[GNU Radio Conference 2026](https://events.gnuradio.org/event/28/contributions/880/). The
workshop covers [TorchSig](https://github.com/TorchDSP/torchsig) for synthetic
RF dataset generation, [TorchSig Models](https://github.com/TorchDSP/torchsig-models)
for training and inference, and TorchSig's geolocation tools.

A separate CTF challenge is also included. It is not part of the workshop.

## Workshop notebooks

| Notebook | Topic |
| --- | --- |
| `TorchSig-GRCon-2026.ipynb` | TorchSig basics: creating a dataset with `TorchSigIterableDataset`, writing it to disk, reading it back with `StaticTorchSigDataset`, and applying impairments |
| `TorchSig-Models-GRCon-2026.ipynb` | TorchSig Models v1.0.0: configuring generated data, training an IQ classifier with the training API, evaluating it, and reloading it for inference |
| `TorchSig-Geo-GRCon-2026.ipynb` | Geolocation: configuring `TorchSigGeoDataset`, defining transmitter/receiver geometry, generating samples, and plotting received signals |

`geo_example_utils.py` provides the geometry plotting helper used by the geo
notebook, plus range-based and TDOA position-estimation helpers for further
experiments.

The notebooks install their own dependencies and are written to run in
Google Colab. A GPU runtime (e.g. T4) is recommended for the models notebook.

## CTF challenge: Excavation Troubles

`TorchSig-CTF-GRCon-2026.ipynb` is a standalone challenge. You get five
unlabeled SigMF recordings in `captures/`. Classify each one with the official
TorchSig Models v1.0.0 narrowband XCiT checkpoint, then join the first letter
of each predicted class, in capture order, to recover the flag.

The notebook downloads the checkpoint (`xcit_narrowband_v1.0.0.ckpt`) on
first use.

Challenge maintainers can regenerate the captures in a GNU Radio Python
environment with TorchSig installed:

```bash
python generate_captures.py --output-dir captures
```

The script generates seeded signals with TorchSig, writes the complex sample
stream with GNU Radio, and saves SigMF metadata. On systems without GNU Radio,
pass `--writer numpy`.

## Running locally

Create a virtual environment and install the dependencies (`ipykernel` is
included in `requirements.txt`):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name grcon-2026 --display-name "Python (GRCon 2026)"
```

Open the notebooks and select the **Python (GRCon 2026)** kernel:

- **VS Code:** open a notebook, click **Select Kernel** in the top right, and
  choose **Python (GRCon 2026)** (or the `.venv` interpreter directly).
- **JupyterLab:** install it with `pip install jupyterlab`, run `jupyter lab`,
  then pick the kernel from **Kernel → Change Kernel**.

To remove the kernel later:

```bash
jupyter kernelspec uninstall grcon-2026
```

Running the notebooks produces `datasets/`, `runs/` and `lightning_logs/`.
Git ignores these folders, along with model checkpoints and the slide deck.
