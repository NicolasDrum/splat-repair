# Contributing

Thanks for looking into Splat Repair.

## Reporting a problem

Open an issue and include:

- what you did and what you expected,
- the file format and roughly how many splats the scan has,
- your GPU and whether Deep Fill is installed,
- a screenshot of the result.

A small scan that shows the problem helps most. Only attach files you are
allowed to share.

## Sending a change

1. Fork the repository and create a branch.
2. Keep a change to one topic.
3. Run the engine tests and the type check before you open a pull request:

```bash
cd core
for %t in (tests\test_*.py) do .venv\Scripts\python %t
```

On macOS:

```bash
cd core && for t in tests/test_*.py; do .venv/bin/python "$t" || echo "FAILED: $t"; done
```

```bash
cd app
npx tsc --noEmit -p .
```

4. Describe what the change does and how you checked it. For a change to a
   fill, add a before and after picture.

## Automated checks

The contribution workflow runs the engine scripts, TypeScript checks and build
on Windows with the base dependencies. To run the same Python scripts from the
repository root with your project environment active, use
`python core/tests/run.py`. The runner lists failures and the cases skipped by
the engine tests, including optional GPU backends. For the app checks, run
`npm ci`, `npm run typecheck`, `node --test` and `npm run build` from `app`.

On a Windows machine with working graphics, generate a synthetic scene and
run the Electron history regression from the repository root:

```powershell
$scene = Join-Path $env:TEMP ('splat-history-' + [guid]::NewGuid() + '.ply')
& .\core\.venv\Scripts\python.exe core/tests/make_electron_fixture.py $scene
Push-Location app
try {
  npm.cmd run build
  if ($LASTEXITCODE -ne 0) { throw 'Build failed' }
  & .\node_modules\.bin\electron.cmd tests/unexported-edits.mjs $scene
  if ($LASTEXITCODE -ne 0) { throw 'Electron test failed' }
} finally {
  Pop-Location
}
```

By contributing you agree that your contribution is licensed under the Apache
License 2.0, as stated in section 5 of the licence.
