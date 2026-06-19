"""Launch SkellyClicker web server: python -m skellyclicker.api"""

import webbrowser
from threading import Timer

import uvicorn

from skellyclicker.api.app import app

DEFAULT_PORT = 8765


def main() -> None:
	# Open browser after a short delay so the server is listening.
	Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{DEFAULT_PORT}")).start()
	uvicorn.run(app, host="127.0.0.1", port=DEFAULT_PORT, log_level="info")


if __name__ == "__main__":
	main()
