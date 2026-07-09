# skellyclicker

For labelling and training data through DeepLabCut.

[![DOI](https://zenodo.org/badge/955944295.svg)](https://doi.org/10.5281/zenodo.19389888)

## Installation

1. Clone the repository

2. Change the directory to the cloned repository

    - `cd skellyclicker`

3. Create and activate the conda environment:

    ```bash
    conda env create -f skellyclicker_env.yaml
    conda activate skellyclicker
    ```

4. Install the package (pulls in `uvicorn`, `fastapi`, `deeplabcut`, etc.):

    ```bash
    pip install -e .
    ```

5. **Ubuntu only:** install native file dialogs (web UI and legacy GUI):

    ```bash
    sudo apt install zenity python3-tk
    ```

    `zenity` is the primary file browser on Ubuntu. SkellyClicker must run on a machine
    with a graphical desktop session (`DISPLAY` or `WAYLAND_DISPLAY` set). Dialogs open
    on the Ubuntu box where the API runs. If server dialogs are unavailable, the web UI
    falls back to the **browser's file picker** (files are uploaded to the server).

### Troubleshooting

**`ModuleNotFoundError: No module named 'uvicorn'`** (or `pandas`, `fastapi`, etc.)

You are using a Python that does not have SkellyClicker installed. Fix:

```bash
conda activate skellyclicker   # must be active every new terminal
cd /path/to/skellyclicker
pip install -e .
python -m skellyclicker.api
```

Confirm the right interpreter: `which python` should point inside your conda env
(e.g. `.../envs/skellyclicker/bin/python`).

## How To Use

### Web UI (recommended)

1. Activate the environment and ensure dependencies are installed (see **Installation** above).

2. Build the frontend (once, or after UI changes):

    ```bash
    cd frontend && npm install && npm run build && cd ..
    ```

3. Start the web server (opens your browser automatically):

    ```bash
    python -m skellyclicker.api
    ```

4. Follow the workflow: **Videos → DeepLabCut → Labels → Train Network / Full Analysis → Session**.

   File pickers use native Ubuntu dialogs via zenity (or the browser file picker as fallback).
   If dialogs are unavailable at startup, the server logs a warning.

### Legacy Tk UI

1. Activate the environment.

2. Open the legacy GUI:

    - `python -m skellyclicker`

3. Start a new session or load an existing one.
    - When loading a session, look for the `.json` file you saved on a previous session.

4. Label the videos by clicking `load videos` on the first iteration, or `open videos` on subsequent iterations
    - Make sure you save the videos after labelling
    
5. Create a DeepLabCut project or load an existing project if you haven't yet

6. Train the model with the `Train Network` button

7. Click `Analyze Videos` to run the model on videos. If you run the model on the training videos, this will allow you to see the models output in the next round of labelling. 

8. Repeat steps 4-7 until the model performs sufficiently well.
