# ONGOING

Running log of in-progress work. Durable architecture, decisions, and gotchas
are promoted to CLAUDE.md once stable; this file holds only what is currently in
flight.

## RoboLab visuals: table texture + background + vendored asset library

Done (durable notes promoted to the **RoboLab Graphics** section of CLAUDE.md):

- Configurable look on the robolab example: `--table {maple,oak,bamboo,black}`,
  `--background <name>`. Both render in the **raycast preview** (the only path
  that runs on this H200), not just the RTX path: table wood base-color is
  planar/triplanar-mapped onto the fixture box; the dome HDR/EXR is tone-mapped
  and sampled as an equirectangular backdrop on ray miss (was a flat brown box +
  sky gradient). Verified headlessly (maple/home_office default, oak/tv_studio).
- Vendored a curated, fully self-contained asset subset into `assets/` — **no
  code reads `_external/` at runtime** (hard constraint; `_external` may vanish):
  backgrounds (5 default + 8 indoor HDR/EXR), all 4 tables as offline `.usda`,
  objects (apple_01, banana, bowl, mug, rubiks_cube + `object_catalog.json`).
- Config resolvers in `robolab_viz/config.py`: `available_tables/_backgrounds/
  _objects`, `resolve_background`, `work_table_fixture`, `background_dome`,
  `object_asset`. `FixtureConfig.texture_file`/`texture_uv_scale` added.

Lighting: reported to the user (RoboLab = dome HDR as IBL+backdrop + a sphere
key light @5000 + optional distant lights); **no lighting changes made** per
the user's instruction.

Possible follow-ups (not requested yet):

- Place vendored objects (apple/banana/rubiks/bowl/mug) into a demo scene — they
  are vendored + cataloged but not yet rendered in any example. Preview-rendering
  a static USD object would reuse the new texture path (load mesh + base color).
- Preview lighting parity: add the dome as ambient + the sphere as a point light
  in the raycast shader (currently one hardcoded directional key + flat ambient).
