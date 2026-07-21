# skellyclicker

For labelling and training data through DeepLabCut.

[![DOI](https://zenodo.org/badge/955944295.svg)](https://doi.org/10.5281/zenodo.19389888)

## Installation

1. Clone the repository

2. Change into the cloned repository:

	```bash
	cd skellyclicker
	```

3. Create and activate the conda environment:

	```bash
	conda env create -f skellyclicker_env.yaml
	conda activate skellyclicker
	```

4. Install the package:

	```bash
	pip install -e .
	```

5. Build the frontend (once, or after UI changes):

	```bash
	cd frontend && npm install && npm run build && cd ..
	```

6. **Ubuntu only:** install native file dialogs:

	```bash
	sudo apt install zenity python3-tk
	```

	SkellyClicker should run on a machine with a graphical desktop session
	(`DISPLAY` or `WAYLAND_DISPLAY` set). File pickers use zenity when available;
	otherwise the web UI falls back to the browser file picker.

## Starting SkellyClicker (new terminal)

Every time you open a new terminal:

```bash
conda activate skellyclicker
cd /path/to/skellyclicker
python -m skellyclicker.api
```

Replace `/path/to/skellyclicker` with your clone path. The server opens the web UI
in your browser. Stop it with `Ctrl+C`.

## How To Use

### Web UI (recommended)

1. Start the server (see **Starting SkellyClicker** above).

2. Follow the workflow: **Videos → DeepLabCut → Labels → Train Network / Full Analysis → Session**.

### Legacy Tk UI

1. Activate the environment and change into the repo.

2. Open the legacy GUI:

	```bash
	python -m skellyclicker
	```

3. Start a new session or load an existing one (look for a previously saved `.json` session file).

4. Label videos with **load videos** (first time) or **open videos** (later), and save after labelling.

5. Create or load a DeepLabCut project.

6. Train with **Train Network**.

7. Run **Analyze Videos**. Analyzing the training videos lets you review model output in the next labelling round.

8. Repeat labelling / train / analyze until the model is good enough.
