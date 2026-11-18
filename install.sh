#!/bin/sh
# Installs Splat Repair into this folder on macOS or Linux. Nothing is
# written outside it.
#
#   ./install.sh         editor with Fast Fill
#   ./install.sh deep    also Deep Fill (NVIDIA GPU with CUDA, not on macOS)
#
# Safe to run again: it only adds what is missing and rebuilds the app.
set -e
cd "$(dirname "$0")"

fail() {
  echo
  echo "The installation did not finish. The lines above show what went wrong."
  exit 1
}

if ! command -v node >/dev/null 2>&1; then
  echo "Node.js is missing. Install version 22 or newer from https://nodejs.org and run this again."
  fail
fi

PY=""
for v in 3.12 3.11 3.13 3.10; do
  if command -v "python$v" >/dev/null 2>&1; then PY="python$v"; break; fi
done
if [ -z "$PY" ] && command -v python3 >/dev/null 2>&1 &&
   python3 -c "import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)" 2>/dev/null; then
  PY="python3"
fi
if [ -z "$PY" ]; then
  echo "Python 3.10 to 3.13 is missing. Install it from https://www.python.org and run this again."
  fail
fi

echo
echo "[1/3] Engine (Python)"
if [ ! -x "core/.venv/bin/python" ]; then
  "$PY" -m venv "core/.venv" || fail
fi
"core/.venv/bin/python" -m pip install --quiet --upgrade pip || fail
"core/.venv/bin/python" -m pip install --quiet -r "core/requirements.txt" || fail

echo
echo "[2/3] App (Electron)"
cd app
npm install || fail
# npm writes package-lock.json with the CRLF line endings of package.json.
# When that is the only change, put the committed file back, so the next
# "git pull" in update.sh is not refused over a file nobody edited.
if [ -d ../.git ] && git diff --quiet --ignore-cr-at-eol -- package-lock.json 2>/dev/null; then
  git checkout -- package-lock.json 2>/dev/null
fi
# The Electron runtime is downloaded on first use; do it now so start.sh finds it.
npx electron --version || fail
npm run build || fail
cd ..

echo
echo "[3/3] Deep Fill"
if [ "$1" = "deep" ]; then
  if [ "$(uname -s)" = "Darwin" ]; then
    echo "Skipped. Deep Fill needs an NVIDIA GPU with CUDA, which macOS does not have."
  else
    "core/.venv/bin/python" "core/setup_deep.py" || fail
  fi
else
  echo "Skipped. Run \"./install.sh deep\" later if you have an NVIDIA GPU and want Deep Fill."
fi

echo
echo "Done. Start Splat Repair with app/start.sh"
