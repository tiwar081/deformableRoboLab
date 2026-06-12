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

### 1. Populate External Sources

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

### 3. Run The Local Example

```bash
python -m examples minimal_cable_franka --viewer usd --device cuda:0
```

For a CPU smoke test that writes a USD file:

```bash
python -m examples minimal_cable_franka --viewer usd --device cpu --num-frames 1 --output-path /tmp/robolab_vbd_smoke.usd --quiet
```
