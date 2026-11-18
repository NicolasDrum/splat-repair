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

By contributing you agree that your contribution is licensed under the Apache
License 2.0, as stated in section 5 of the licence.
