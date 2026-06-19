"""URDF -> USD conversion via Isaac Sim's importer, cached under ``assets/robots/``.

Why USD-from-URDF instead of a hand-authored robot USD: the physics examples
simulate ``fr3_franka_hand.urdf`` (Newton asset pack). Converting that same
URDF with Isaac Sim's importer guarantees the rendered robot geometry matches
the simulated robot exactly (same links, same meshes, same frames) while
producing Isaac-native materials (MDL/OmniPBR) — the same quality RoboLab's
own assets go through.

The conversion boots a minimal renderer-less kit kernel and is run in a
*subprocess*: on GPUs without RT cores (H100/H200) kit's graphics device dies
at startup, but USD authoring is CPU-side and completes anyway; isolating it
in a child process keeps the GPU state of the simulation process clean.

The output is cached: examples call :func:`ensure_robot_usd` which converts
once and reuses ``assets/robots/<name>/<name>.usd`` afterwards, so a normal
run never needs kit at all.

CLI: ``python -m robolabViz.robot_usd <urdf_path> <output_usd_path>``
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def ensure_robot_usd(urdf_path: Path | str, name: str = "fr3_franka_hand", robots_dir: Path | None = None) -> Path:
    """Return the cached USD conversion of ``urdf_path``, converting on first use."""
    if robots_dir is None:
        robots_dir = Path(__file__).resolve().parent.parent / "assets" / "robots"
    out_path = robots_dir / name / f"{name}.usd"
    if out_path.exists():
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[robolabViz] Converting {urdf_path} -> {out_path} (one-time, via Isaac Sim URDF importer)...")
    env = dict(os.environ, OMNI_KIT_ALLOW_ROOT="1")
    result = subprocess.run(
        [sys.executable, "-m", "robolabViz.robot_usd", str(urdf_path), str(out_path)],
        cwd=Path(__file__).resolve().parent.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if not out_path.exists():
        raise RuntimeError(
            "URDF -> USD conversion failed.\n"
            f"stdout (tail):\n{result.stdout[-3000:]}\n"
            f"stderr (tail):\n{result.stderr[-3000:]}"
        )
    fix_dae_visual_orientation(out_path, urdf_path)
    return out_path


def fix_dae_visual_orientation(robot_usd_path: Path | str, urdf_path: Path | str) -> int:
    """Undo the importer's wrong up-axis correction on Z-up Collada visual meshes.

    Isaac Sim's asset converter applies a Y-up -> Z-up rotation (+90 deg about
    X, authored as ``xformOp:orient = (0.70711, 0.70711, 0, 0)`` on each visual
    mesh prim) assuming Collada's default Y-up. The Franka visual DAEs declare
    ``<up_axis>Z_UP</up_axis>``, so the correction mis-rotates every visual by
    90 deg (at q=0 the closed fingers end up sideways). All FR3 URDF visual
    origins are identity, so the correct mesh-local orient is identity.

    Edits the conversion's base layer in place; returns the number of mesh
    prims fixed. Idempotent.
    """
    import re
    import xml.etree.ElementTree as ET

    from pxr import Gf, Usd, UsdGeom

    robot_usd_path = Path(robot_usd_path)
    base_layer = robot_usd_path.parent / "configuration" / f"{robot_usd_path.stem}_base.usd"
    if not base_layer.exists():
        raise FileNotFoundError(f"Converted robot base layer not found: {base_layer}")

    # link -> True if its visual mesh is a Z-up DAE
    urdf_dir = Path(urdf_path).parent
    zup_links: set[str] = set()
    root = ET.parse(urdf_path).getroot()
    for link in root.iter("link"):
        for visual in link.iter("visual"):
            for mesh in visual.iter("mesh"):
                fname = mesh.get("filename", "")
                if not fname.lower().endswith(".dae"):
                    continue
                rel = re.sub(r"^package://[^/]+/", "", fname)
                candidates = [urdf_dir / rel, urdf_dir.parent / rel]
                for c in candidates:
                    if c.exists():
                        if "<up_axis>Z_UP</up_axis>" in c.read_text(errors="ignore"):
                            zup_links.add(link.get("name"))
                        break

    stage = Usd.Stage.Open(str(base_layer))
    fixed = 0
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if prim.GetTypeName() != "Mesh" or not path.startswith("/visuals/"):
            continue
        link_name = path.split("/")[2]
        if link_name not in zup_links:
            continue
        for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                current = op.Get()
                if current is not None and abs(float(current.GetReal()) - 1.0) > 1e-6:
                    op.Set(type(current)(1.0, 0.0, 0.0, 0.0))
                    fixed += 1
    if fixed:
        stage.GetRootLayer().Save()
    print(f"[robolabViz] Fixed DAE up-axis orient on {fixed} visual mesh prims in {base_layer.name}")
    return fixed


def _convert(urdf_path: str, dest_path: str) -> None:
    """Run inside the kit subprocess: boot a minimal kernel and import the URDF."""
    site = str(Path(os.__file__).parent / "site-packages")

    from omni.kit_app import KitApp

    app = KitApp()
    app.startup(
        [
            "--no-window",
            "--/app/window/enabled=false",
            "--/app/asyncRendering=false",
            "--/renderer/enabled=false",
            "--/app/fastShutdown=true",
            "--/crashreporter/enabled=false",
            "--/telemetry/enableAnonymousData=false",
            "--/app/settings/persistent=false",
            f"--ext-folder={site}/isaacsim/exts",
            f"--ext-folder={site}/isaacsim/extscache",
            f"--ext-folder={site}/isaacsim/extsPhysics",
            "--enable",
            "omni.usd",
            "--enable",
            "omni.kit.commands",
            "--enable",
            "isaacsim.asset.importer.urdf",
        ]
    )

    import omni.kit.app

    for _ in range(4):
        omni.kit.app.get_app().update()

    import omni.kit.commands
    from isaacsim.asset.importer.urdf import _urdf

    cfg = _urdf.ImportConfig()
    # Keep every URDF link as its own prim (no merging): the puppeteer poses
    # links by name from an uncollapsed Newton FK model.
    cfg.merge_fixed_joints = False
    cfg.fix_base = True
    cfg.import_inertia_tensor = False
    cfg.distance_scale = 1.0
    cfg.make_default_prim = True
    cfg.self_collision = False

    status, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=urdf_path,
        import_config=cfg,
        dest_path=dest_path,
    )
    print(f"[robolabViz] import status={status} prim={prim_path}")

    from pxr import Usd

    stage = Usd.Stage.Open(dest_path)
    if stage is None or stage.GetDefaultPrim() is None or not stage.GetDefaultPrim().GetChildren():
        raise RuntimeError(f"URDF import produced an empty stage at {dest_path}.")
    app.shutdown()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m robolabViz.robot_usd <urdf_path> <output_usd_path>")
    _convert(sys.argv[1], sys.argv[2])
