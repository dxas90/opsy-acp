"""Thin shim — run the agent via `uv run main.py` or `uv run opsy-acp`."""

from opsy_acp.server import default_agent as main

if __name__ == "__main__":
    main()
