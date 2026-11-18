#!/bin/sh
# Updates Splat Repair to the newest version and rebuilds it.
# Works for a copy made with "git clone". If you downloaded a ZIP, download
# the new ZIP instead and run install.sh in it.
cd "$(dirname "$0")" || exit 1

if ! command -v git >/dev/null 2>&1; then
  echo "Git is missing. Install it from https://git-scm.com or download the new ZIP from GitHub."
  exit 1
fi
if [ ! -d ".git" ]; then
  echo "This folder is not a git clone. Download the new ZIP from GitHub and run install.sh in it."
  exit 1
fi

if ! git pull --ff-only; then
  echo
  echo "Could not update. If you changed files yourself, commit or undo those changes first."
  exit 1
fi

exec ./install.sh "$@"
