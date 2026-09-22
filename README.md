# TorchSig Workshop: GRCon 2026

Materials for the TorchSig workshop at
[GNU Radio Conference 2026](https://events.gnuradio.org/event/28/contributions/880/). The
workshop covers [TorchSig](https://github.com/TorchDSP/torchsig) for synthetic
RF dataset generation, [TorchSig Models](https://github.com/TorchDSP/torchsig-models)
for training and inference, and TorchSig's geolocation tools.

A separate CTF challenge is also included. It is not part of the workshop.

You can also run our notebooks in [Google Colab](https://drive.google.com/drive/folders/1Wg4y3GPt2FojUfpXB6s8n6Uztsl-Qxc-?usp=sharing).

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

## Running locally

These steps work on Windows, macOS and Linux. Where a command differs, each
OS has its own version.

Set up the environment with **uv** (recommended) or with **venv and pip**. Both
give you the same `.venv` folder, and the VS Code and JupyterLab steps below
work with either.

### Prerequisites

- **Git.** `requirements.txt` installs TorchSig Models straight from GitHub,
  so the installer needs `git` on your `PATH`. On Windows, install
  [Git for Windows](https://git-scm.com/download/win) and reopen your terminal.
- **Python 3.10 or newer.** You only need this for the venv and pip route;
  uv downloads Python for you. Check with `python3 --version` on
  macOS/Linux or `py --version` on Windows.
  - Windows: install from [python.org](https://www.python.org/downloads/).
    It includes the `py` launcher.
  - macOS: install from python.org or with Homebrew (`brew install python`).
  - Debian/Ubuntu: `sudo apt install python3 python3-venv python3-pip`.

In every option below, start by opening a terminal (PowerShell on Windows)
and changing into the repository folder:

```bash
git clone https://github.com/TorchDSP/grcon2026.git
cd grcon2026
```

### Option 1: Set up with uv (recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package and environment
manager. It installs dependencies much faster than pip and can download the
right Python version itself.

1. Check whether uv is installed:

   ```bash
   uv --version
   ```

   If this prints a version, go to step 2. If not, install it.

   macOS/Linux:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   Windows (PowerShell):

   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

   You can also install it with Homebrew (`brew install uv`), WinGet
   (`winget install --id=astral-sh.uv -e`) or pip (`pip install uv`).

   Then **close and reopen your terminal** so `uv` is on your `PATH`, change
   back into the repository folder, and run `uv --version` again.

2. Create a virtual environment. uv downloads Python 3.12 if you don't
   already have it:

   ```bash
   uv venv --python 3.12
   ```

3. Install the dependencies. `requirements.txt` includes `ipykernel` and
   `jupyterlab`:

   ```bash
   uv pip install -r requirements.txt
   ```

   uv installs into `.venv` in the current folder, so you don't need to
   activate it. If a different virtual environment or conda environment is
   already active, uv installs into that one instead. Run `deactivate` (or
   `conda deactivate`) first.

4. Register the environment as a Jupyter kernel:

   ```bash
   uv run python -m ipykernel install --user --name grcon-2026 --display-name "Python (GRCon 2026)"
   ```

5. Optional: activate the environment so plain `python` and `jupyter`
   commands use it. You can also prefix commands with `uv run`, as in
   `uv run jupyter lab`.

   | Shell | Command |
   | --- | --- |
   | macOS/Linux (bash, zsh) | `source .venv/bin/activate` |
   | Windows PowerShell | `.venv\Scripts\Activate.ps1` |
   | Windows Command Prompt | `.venv\Scripts\activate.bat` |
   | Windows Git Bash | `source .venv/Scripts/activate` |

   If PowerShell says running scripts is disabled, run
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again.

### Option 2: Set up with venv and pip

1. Create a virtual environment.

   macOS/Linux:

   ```bash
   python3 -m venv .venv
   ```

   Windows (PowerShell or Command Prompt):

   ```powershell
   py -m venv .venv
   ```

2. Activate it with the command for your shell from the table in Option 1,
   step 5. Your prompt shows `(.venv)` once it's active. From here on,
   `python` refers to the virtual environment on every OS.

3. Install the dependencies:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

4. Register the environment as a Jupyter kernel:

   ```bash
   python -m ipykernel install --user --name grcon-2026 --display-name "Python (GRCon 2026)"
   ```

### About the PyTorch download

Either option installs PyTorch, which is a large download. On Linux the
default PyTorch wheel includes CUDA libraries. For other builds (CPU-only, a
specific CUDA version), install `torch` first by following
[pytorch.org](https://pytorch.org/get-started/locally/), then install
`requirements.txt`. With uv, run the pytorch.org `pip install` command as
`uv pip install`.

Then follow either the VS Code or the JupyterLab steps below.

### Before you run a notebook

Each notebook starts with setup cells written for Google Colab. When you run
the notebooks locally, skip them:

- **`!pip install ...` cells:** skip these. Your environment already has
  everything from `requirements.txt`, and
  `!pip install -q torchsig` would replace the pinned `torchsig==2.2.0` with
  the latest release.
- **`!curl ... grcon26-assets.zip` cells** (CTF and Geo notebooks): skip these.
  The repository already contains `captures/` and `geo_example_utils.py`.
  These cells also use `unzip`, `mv` and `rm`, which Windows doesn't have.

Start from the first `import` cell.

### Option A: VS Code

1. Install the **Python** and **Jupyter** extensions from the Extensions view.
2. Open the repository folder with **File → Open Folder...** (**File →
   Open...** on macOS).
3. Open a notebook, such as `TorchSig-GRCon-2026.ipynb`.
4. Click **Select Kernel** in the top right of the notebook.
5. Choose **Jupyter Kernel... → Python (GRCon 2026)**. You can also choose
   **Python Environments... → .venv**.
6. Click in the first `import` cell, open the **...** menu on its toolbar,
   and choose **Execute Cell and Below**. You can also step through cells
   with **Shift+Enter**. Don't use **Run All**, because it runs the setup
   cells described above.

### Option B: JupyterLab

1. Start JupyterLab from the repository root. With the virtual environment
   activated:

   ```bash
   jupyter lab
   ```

   Or, with uv and no activation:

   ```bash
   uv run jupyter lab
   ```

2. JupyterLab opens in your browser. If it doesn't, open the
   `http://localhost:8888/lab?token=...` link printed in the terminal.
3. Double-click a notebook in the file browser on the left.
4. Choose **Kernel → Change Kernel...** and select **Python (GRCon 2026)**.
5. Select the first `import` cell and choose **Run → Run Selected Cell and
   All Below**, or step through cells with **Shift+Enter**.
6. When you're done, stop the server with **File → Shut Down**. You can also
   press **Ctrl+C** in the terminal and answer `y` (on macOS too, it's
   Ctrl, not Cmd).

### Cleaning up

To remove the registered kernel, activate the virtual environment and run:

```bash
jupyter kernelspec uninstall grcon-2026
```

Or, with uv:

```bash
uv run jupyter kernelspec uninstall grcon-2026
```

To remove the environment itself, delete the `.venv` folder.

Running the notebooks produces `datasets/`, `runs/` and `lightning_logs/`.
Git ignores these folders, along with model checkpoints and the slide deck.

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