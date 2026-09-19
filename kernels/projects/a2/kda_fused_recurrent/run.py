"""Canonical standalone entry for the A2 KDA decode (fused recurrent) unit."""
from pathlib import Path

from _unit_runner import main


if __name__ == "__main__":
    raise SystemExit(main(Path(__file__).resolve().parent))
