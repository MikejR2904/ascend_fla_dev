"""Run the GDN-2 decode short-conv unit with the canonical unit helper."""
from pathlib import Path

from _unit_runner import main


if __name__ == "__main__":
    raise SystemExit(main(Path(__file__).resolve().parent))
