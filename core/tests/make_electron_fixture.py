"""Generate the small plane used by the Electron history test."""
import argparse
from pathlib import Path

from test_server import make_ply


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="an unused .ply destination")
    args = parser.parse_args()
    if args.output.exists() or args.output.suffix.lower() != ".ply":
        parser.error("output must be an unused .ply destination")
    make_ply(args.output, n=24000)
    print(args.output.resolve())
