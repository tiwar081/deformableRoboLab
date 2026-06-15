# Asset Attribution

The fixture, material, and background assets in this directory are copied from
NVIDIA's RoboLab project (https://github.com/NVLabs/RoboLab), licensed under
Apache-2.0 (SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION &
AFFILIATES).

They are vendored here so this repository renders the same table / lighting /
scene as RoboLab's `examples/run_recorded.py` without depending on the
`_external/` checkout:

- `fixtures/franka_table.usd` (+ `fixtures/Props/instaceable_meshes.usd`) —
  the pedestal the Franka is mounted on.
- `fixtures/table_maple.usda` — the work table. The `.usda` variant is used
  (instead of the binary `.usd`) because it references the local
  `../materials` MDL copies rather than omniverse-content-production S3 URLs,
  so rendering works offline.
- `materials/...` — the MDL materials + textures referenced by
  `table_maple.usda` (Oak, Bamboo, Walnut_Planks, RustedMetal, Plastic_ABS,
  2023_1 vMaterials plastics). `OmniPBR.mdl` is not vendored; it ships with
  every Omniverse/Isaac Sim install and resolves from kit's MDL search paths.
- `backgrounds/home_office.exr` — the dome-light HDR used as the default
  RoboLab background.

`robots/` contains USD conversions of robot URDFs generated locally by
`python -m robolab_viz.robot_usd` (source URDF: Newton's
`franka_emika_panda` asset pack, BSD-licensed by Franka Robotics).
