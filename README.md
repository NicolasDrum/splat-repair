# Splat Repair

An editor for Gaussian splats with a repair brush. You paint over what should
go, and the tool rebuilds the surface underneath: geometry, colour and
view-dependent appearance.

| Scan | After Fast Fill |
|---|---|
| ![A beach hut scan with a red cross on a sign](docs/before.jpg) | ![The same sign with the cross painted out](docs/after.jpg) |

I built this in eight days and filmed it:
[I built Photoshop for Gaussian Splats in 8 days](https://www.youtube.com/watch?v=qwjmxt8kfl0).

## What works and what doesn't

- **Flat and gently curved surfaces** fill in about a second. Walls, signs,
  floors, stickers and small handles are the easy cases.
- **Strong repeating relief** is hard. On the sole of a shoe the fill often
  shows and needs patching by hand with the Clone Stamp.
- This is an early release from one person. Expect rough edges, and save
  copies of your scans.

## Tools

| Tool | Key | What it does |
|---|---|---|
| View | `V` | Camera only |
| Brush | `B` | Paints a 3D selection along the stroke |
| Lasso | `L` | Freehand selection, surface only or through the object |
| Rectangle | `R` | Box selection |
| Clone Stamp | `S` | Alt+click sets the source, painting stamps it |
| Move | `G` | Transforms the selection |
| Clean up | | Finds floaters, crops with a box |

On a selection you can delete, fill or export.

- **Fast Fill** is geometric. It fits the surface around the hole and copies
  whole splats from nearby, so scale, rotation and spherical harmonics stay
  consistent. It runs on the CPU in about 0.3 to 1.5 s on a scan with 550,000
  splats.
- **Deep Fill** renders a view, lets an inpainting model (LaMa) fill the
  picture and lifts the result back into splats. It needs an NVIDIA GPU and
  takes around 20 s on an RTX 4090.
- **Fill from Photoshop** (optional) sends the rendered view to your own
  Adobe Photoshop installation and uses its Content-Aware or Generative Fill.

Splat Repair reads and writes `.ply`, `.splat` and `.spz`. PLY is the working
format and round-trips bit-exact.

## Requirements

- Windows 10 or 11, or macOS on Apple Silicon. Linux is not tested.
- [Node.js](https://nodejs.org) 22 or newer.
- Python 3.10 to 3.13.
- For Deep Fill: an NVIDIA GPU, a CUDA toolkit and the MSVC build tools
  (gsplat compiles on first use). About 3 GB of downloads.

On macOS the editor, Fast Fill, the Clone Stamp and the selection tools work.
Deep Fill and the hidden-debris part of Clean up need an NVIDIA GPU, so they
are not available there; Clean up still finds splats cut off from the rest.

## Install

1. Install [Git](https://git-scm.com), [Node.js](https://nodejs.org) and
   [Python](https://www.python.org) if you do not have them.
2. Get the code and run the installer:

```bash
git clone https://github.com/NicolasDrum/splat-repair.git
```

```bash
cd splat-repair
```

```bash
install.cmd
```

3. Start the app with `app\start.cmd`.

Everything is installed inside the `splat-repair` folder. To remove Splat
Repair, delete the folder.

For Deep Fill (PyTorch, gsplat and LaMa), run the installer with `deep`. You
can do that at any time:

```bash
install.cmd deep
```

During the Deep Fill setup pip prints dependency conflicts for
`simple-lama-inpainting`. That is expected; the setup ends with "Deep Fill is
ready" when everything works.

Without Git you can also download the ZIP from the green Code button on
GitHub, unpack it and run `install.cmd` in the unpacked folder.

`app\create-shortcut.ps1` puts a shortcut on the desktop:

```bash
powershell -ExecutionPolicy Bypass -File app\create-shortcut.ps1
```

### macOS

The same steps with the shell scripts: clone the repository, then run the
installer in the `splat-repair` folder:

```bash
./install.sh
```

Start the app with `app/start.sh`.

## Update

```bash
update.cmd
```

On macOS:

```bash
./update.sh
```

This pulls the newest version and rebuilds the app. Add `deep` if you use
Deep Fill and the release notes mention changed requirements.

<details>
<summary>Manual install</summary>

```bash
cd core
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

```bash
cd app
npm install
npm run build
npx electron .
```

```bash
core\.venv\Scripts\python.exe core\setup_deep.py
```

</details>

## Develop

```bash
cd app
npm run dev
```

Run the engine tests (each file is a plain script):

```bash
cd core
for %t in (tests\test_*.py) do .venv\Scripts\python %t
```

On macOS:

```bash
cd core && for t in tests/test_*.py; do .venv/bin/python "$t" || echo "FAILED: $t"; done
```

## How it is built

```
app/      Electron + PlayCanvas (WebGPU): the interface
  electron/      main process, preload, engine process handling
  src/engine/    viewer, selection, volumes, document
  src/ui/        menus, history, view cube
core/     Python engine: formats, repair, document, server
```

- The interface and the engine talk over the child process pipe. There is no
  HTTP server and no open port.
- Both sides read the same file, so splat index 12345 means the same splat in
  the viewer and in the engine. An edit crosses the pipe as a few thousand
  numbers.
- Indices are stable. The loaded file keeps its range, new splats are
  appended, and edits flip `alive` flags, which makes undo cheap.

## Licence

Apache License 2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). Third-party
components and their licences are listed in
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

Deep Fill downloads the LaMa model on first use. Its authors publish code and
weights under Apache 2.0; the model was trained on the Places2 dataset, which
has its own terms. Check them if you use Deep Fill commercially.

Adobe and Photoshop are trademarks of Adobe Inc. Splat Repair is an
independent project and is not affiliated with or endorsed by Adobe.

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

Made by [Nicolas Drum](https://nicolasdrum.com).
