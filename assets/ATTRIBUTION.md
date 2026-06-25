# Asset Attribution

The fixture, material, background, and object assets in this directory are
copied from NVIDIA's RoboLab project (https://github.com/NVLabs/RoboLab),
licensed under Apache-2.0 (SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA
CORPORATION & AFFILIATES).

They are vendored here so this repository renders the RoboLab look (table /
lighting / scene / props of `examples/run_recorded.py`) **without ever reading
the `_external/` checkout at runtime** — `_external/RoboLab` may be deleted and
nothing here breaks. Only a curated subset is vendored.

- `fixtures/franka_table.usd` (+ `fixtures/Props/instaceable_meshes.usd`) —
  the pedestal the Franka is mounted on.
- `fixtures/table_{maple,oak,bamboo,black}.usda` — the four work tables
  (`--table`). The `.usda` variants are used (not the binary `.usd`) because
  they reference the local `../materials` MDL copies rather than
  omniverse-content-production S3 URLs, so rendering works offline. The four
  share identical geometry and differ only in the `top` slab's bound material
  (Walnut_Planks / Oak / Bamboo / Black_Matte); oak/bamboo/black were derived
  from `table_maple.usda` by swapping that one binding (RoboLab ships oak/black
  only as S3-referencing binary `.usd`).
- `materials/...` — the MDL materials + base-color/normal/ORM textures
  referenced by the tables (Oak, Bamboo, Walnut_Planks, RustedMetal,
  Plastic_ABS, 2023_1 vMaterials plastics + Carpaint). The raycast preview
  planar-maps the wood `*_BaseColor.png`; the RTX path uses the MDLs.
  `OmniPBR.mdl` is not vendored; it ships with every Omniverse/Isaac Sim
  install and resolves from kit's MDL search paths.
- `backgrounds/default/*.{exr,hdr}` and `backgrounds/indoors/*.hdr` — a curated
  set of dome-light environment maps (`--background`): the 5 RoboLab defaults
  (incl. `home_office.exr`, the scene default) + 8 indoor scenes. RoboLab's
  `outdoors/` folder has no usable `.hdr`/`.exr` (only PNG previews), so no
  outdoor maps are vendored.
- `objects/{objaverse/apple_01,ycb/banana,ycb/bowl,ycb/mug,hot3d/rubiks_cube}.usd`
  (+ each one's `textures/`) — curated RoboLab scene props for future demos,
  self-contained (mesh + local base-color texture; `OmniPBR.mdl` / `gltf/pbr.mdl`
  resolve from kit). `objects/object_catalog.json` is a trimmed copy of
  RoboLab's catalog (name / usd_path / dims / class) for these objects.

`robots/` contains USD conversions of robot URDFs generated locally by
`python -m robolabViz.robot_usd` (source URDF: Newton's
`franka_emika_panda` asset pack, BSD-licensed by Franka Robotics).
`robots/franka_panda_isaacsim/` is the NVIDIA Isaac Sim 5.0 Franka Panda USD
asset downloaded from
`Assets/Isaac/5.0/Isaac/Robots/FrankaRobotics/FrankaPanda/` in NVIDIA's
`omniverse-content-production` S3 bucket. The included
`franka_alt_fingers_quality.usda` wrapper selects the asset's
`Gripper=AlternateFinger` and `Mesh=Quality` variants. NVIDIA ships
`franka-LICENSE.txt` with this asset.
