"""Standalone reference and bounded diagnostics; hardware acceptance is separate."""
from pathlib import Path

from _unit_runner import main


if __name__ == '__main__':
    raise SystemExit(main(Path(__file__).resolve().parent))
