#!/bin/sh
# Starts Splat Repair on macOS or Linux. Builds first if nothing is built yet,
# or if a build input is newer than the last build (the same check as
# start.cmd). The app runs on its own; this script returns at once.
cd "$(dirname "$0")" || exit 1

build=0
for f in dist/electron/main.mjs dist/electron/preload.cjs dist/renderer/index.html; do
  [ -f "$f" ] || build=1
done
# Without node there is no way to check or build: start the existing build.
if [ "$build" = 0 ] && command -v node >/dev/null 2>&1; then
  node -e "const fs=require('fs'),p=require('path');const m=f=>fs.statSync(f).mtimeMs;const t=Math.min(m('dist/electron/main.mjs'),m('dist/electron/preload.cjs'),m('dist/renderer/index.html'));const newer=f=>fs.statSync(f).isDirectory()?fs.readdirSync(f).some(e=>newer(p.join(f,e))):m(f)>t;process.exit(['src','electron','public','package.json','package-lock.json','vite.config.ts','build.mjs','tsconfig.json'].some(f=>fs.existsSync(f)&&newer(f))?1:0)" || build=1
fi

if [ "$build" = 1 ]; then
  echo "Building Splat Repair ..."
  if ! npm run build; then
    echo
    echo "Build failed."
    exit 1
  fi
fi

nohup ./node_modules/.bin/electron . "$@" >/dev/null 2>&1 &
