"""Run the standalone engine checks and summarize failures and skipped cases."""
from pathlib import Path
import os
import re
import subprocess
import sys


def run_tests(scripts):
    """Return a shell exit code after running every script."""
    if not scripts:
        print("No engine tests found.", file=sys.stderr)
        return 1
    failures, skips = [], []
    env = {**os.environ, "PYTHONUTF8": "1"}
    for script in scripts:
        print(f"\n--- {script.name} ---", flush=True)
        result = subprocess.run(
            [sys.executable, str(script.resolve())],
            cwd=script.parent.parent, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            encoding="utf-8", errors="replace", check=False,
        )
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        skips.extend(f"{script.name}: {line.strip()}" for line in result.stdout.splitlines()
                     if re.search(r"\bskip(?:ped|ping)?\b", line, re.IGNORECASE))
        if result.returncode:
            failures.append(f"{script.name} (exit {result.returncode})")

    print(f"\n{len(scripts) - len(failures)}/{len(scripts)} engine scripts exited successfully.")
    for failure in failures:
        print(f"FAILED: {failure}")
    if skips:
        print("Skipped cases reported by the scripts:")
        for skip in skips:
            print(f"  {skip}")
    return int(bool(failures))


if __name__ == "__main__":
    sys.exit(run_tests(sorted(Path(__file__).parent.glob("test_*.py"))))
