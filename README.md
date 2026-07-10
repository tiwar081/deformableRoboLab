# RoboLab_VBD

This repo contains local VBD examples that use Newton together with RoboLab.

## Environment Setup

Use Python 3.11. RoboLab requires Python `>=3.11`, and the tested environment
uses CUDA 12.8 PyTorch wheels.

The top-level `requirements.txt` builds the compatible single-venv setup used
for this repo:

- Newton is installed editable from `_external/newton`.
- RoboLab is installed editable from `_external/RoboLab`.
- Newton's USD/VBD workflow is supported.
- Newton's full `examples` extra is intentionally excluded because it requires
  `pyglet>=2.1.6`, while `isaaclab==2.2.0` requires `pyglet<2`.
- Newton's full `importers` extra is intentionally excluded because it requires
  `trimesh>=4.6.8`, while IsaacSim 5.0 pins `trimesh==4.5.1`.

### 1. Populate External Sources (FOR DEVELOPERS)

If `_external/newton` and `_external/RoboLab` are not already present, clone
them first:

```bash
mkdir -p _external
git clone https://github.com/newton-physics/newton.git _external/newton
git -C _external/newton checkout 2a1d4215
git clone https://github.com/NVlabs/RoboLab.git _external/RoboLab
git -C _external/RoboLab checkout 7d45d74
```

### 2. Create The Virtualenv

Install with `uv`:

```bash
uv venv .venv --python 3.11 --seed
source .venv/bin/activate
uv pip install --python .venv/bin/python --link-mode copy "setuptools<81" "wheel<0.45"
uv pip install \
  --python .venv/bin/python \
  --prerelease allow \
  --index-strategy unsafe-best-match \
  --torch-backend cu128 \
  --index https://pypi.nvidia.com \
  --index https://download.pytorch.org/whl/cu128 \
  --link-mode copy \
  --no-build-isolation-package flatdict \
  -r requirements.txt
```

Verify the environment:

```bash
python -m pip check
python -c "import warp, newton, robolab, torch; print(warp.__version__, newton.__version__, torch.__version__)"
```

IsaacSim/IsaacLab prompts for the NVIDIA Omniverse EULA on first import. After
reading and accepting the EULA, run Isaac commands with:

```bash
OMNI_KIT_ACCEPT_EULA=yes python -c "import isaaclab; print('isaaclab ok')"
```

## Running the demos

Each Franka manipulation demo is a single file in `examples/` (list them with
`python -m examples --list`). The `--output-style` flag selects how a run is rendered
(all artifacts land in `outputs/<robot>/<name>/`):

```bash
# mp4 (default): lightweight video — simulation.mp4 with the over-shoulder-left +
#   wrist cameras side by side (flat shading, no HDRI decode; table texture kept).
python -m examples cable_soft_franka --device cuda:0

# mp4_advanced: the RoboLab look — HDRI-lit PBR ray tracing (shadows, textures, AA) →
#   simulation_advanced.mp4 + frames/ stills + wrist_coverage.json. Per-demo
#   customization via DemoSpec.render.
python -m examples cable_soft_franka --output-style mp4_advanced --device cuda:0

# usd: the lightest — a plain time-sampled Newton USD at outputs/<robot>/<name>/<name>.usd.
python -m examples cable_soft_franka --output-style usd --device cuda:0
```

(Deprecated aliases still accepted: `basic`→`usd`, `scenic`→`mp4_advanced`. The default
style comes from `settings.yaml` `render.style`.)

Add `--test` to run the demo's physics + render assertions. See
[docs/examples.md](docs/examples.md) for the full demo list and flags, and
[docs/robolab-graphics.md](docs/robolab-graphics.md) for the renderer.
