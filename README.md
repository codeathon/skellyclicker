# skellyclicker

For labelling and training data through DeepLabCut.

[![DOI](https://zenodo.org/badge/955944295.svg)](https://doi.org/10.5281/zenodo.19389888)

## Installation

1. Clone the repository

2. Change the directory to the cloned repository

    - `cd skellyclicker`

3. Create a new conda environment from the environment yaml

    - `conda env create -f skellyclicker_env.yaml`

4. **Ubuntu only:** install tkinter for native file dialogs (used by the web UI and legacy GUI):

    ```bash
    sudo apt install python3-tk
    ```

    SkellyClicker must run on a machine with a graphical desktop session (`DISPLAY` or
    `WAYLAND_DISPLAY` set). Dialogs open on the Ubuntu box where the API runs, not inside
    the browser. Headless SSH without X11 forwarding falls back to typing paths manually.

## How To Use

### Web UI (recommended)

1. Activate the environment.

2. Install Python dependencies and build the frontend:

    ```bash
    pip install -e .
    cd frontend && npm install && npm run build && cd ..
    ```

3. Start the web server (opens your browser automatically):

    ```bash
    python -m skellyclicker.api
    ```

4. Follow the workflow: **Videos → DeepLabCut → Labels → Train & Analyze → Session**.

   File pickers use native Ubuntu dialogs via tkinter (absolute paths on the server filesystem).
   If dialogs are unavailable at startup, the server logs a warning and the UI falls back to
   manual path entry.

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
