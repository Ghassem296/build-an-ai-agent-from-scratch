# Python version notes

## Current target: Python 3.12

Development and CI run on Python 3.11 - 3.13. The local virtual environment
(`.venv`) is CPython 3.12.15, managed by `uv`.

`pyproject.toml` keeps `requires-python = ">=3.11"`; this is intentional and
matches the CI matrix in `.github/workflows/tests.yml`.

## Why not 3.15 (yet)

`uv sync --python 3.15` fails while building `fastuuid==0.14.0`, a transitive
dependency of `litellm`. That release ships wheels for cp38 - cp314 only; there
is no cp315 wheel on PyPI, so maturin has to compile the crate from source.

On this machine the compile fails at the link step:

1. `LNK1104: cannot open file 'msvcrt.lib'`. The installed Windows SDK
   (10.0.26100.0) did not contain `msvcrt.lib` anywhere under
   `C:\Program Files (x86)\Windows Kits`, and the MSVC toolset shipped no
   libraries at all (`lib\x64` was missing or empty).
2. Adding `RUSTFLAGS="-C link-arg=/nodefaultlib:msvcrt"` only moved the failure
   to `LNK2001` (unresolved `memcpy`, `mainCRTStartup`, `__CxxFrameHandler3`,
   ...), confirming that the CRT import libraries were absent rather than
   merely missing from `LIB`.

Installing further Visual Studio C++ components did not help: the toolset still
provided `cl.exe` and `link.exe` without the libraries. Visual Studio and Unity
were later uninstalled from this machine, so the MSVC toolchain is no longer
available here.

## When to revisit

Revisit 3.15 once `fastuuid` publishes a `cp315` wheel:
<https://pypi.org/project/fastuuid/#files>

Migration at that point:

```bash
uv python pin 3.15
uv sync
python -m unittest discover -s tests -v
```

If a wheel is still missing, Python 3.14 is the safer step: every native
dependency of this project already publishes cp314 wheels.

## Rust note

Rust (`cargo`/`rustc` 1.99.1) is installed independently of Visual Studio.
`cargo check` and `cargo clippy` work without a linker, but `cargo build` needs
either the full MSVC C++ workload or the GNU toolchain:

```bash
winget install BrechtSanders.WinLibs.POSIX.UCRT
rustup toolchain install stable-x86_64-pc-windows-gnu
rustup default stable-x86_64-pc-windows-gnu