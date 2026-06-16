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

## Newton 1.4 VBD-damping change → object-specific re-tune + 4× softer blocks

Root cause (diagnosis): `newton` is editable-installed from `_external/newton`,
which **drifted off the README-pinned commit `2a1d4215`** (pre-absolute-damping)
to `2c242002`. Newton `c1af91d2` "Use absolute VBD damping" reinterprets VBD
damping from stiffness-relative (`D = kd·ke`) to **absolute** units, and
reformulates tet damping into an objective `C=FᵀF` strain-rate metric that no
longer damps rigid rotation. With unchanged authored values the object sim went
under-damped → the swept cable pumped the `cable_soft` block off the table (drift
~1.5 m, fell to z≈−12 m). NOT friction (raising `soft_contact_mu` made it worse).
Decision (user, "S2"): target the new Newton and re-tune our values.

Net change — `examples/cable_soft_franka.py` damping:
- **Object-specific** damping migrated `kd_new = kd_old·k` (kept; verified
  pure-rescale in the kernels, so reproduces the pre-1.4 effective damping):
  cable-shape `kd 20→4e5`, proxy-shape `kd 1e2→5e6`, rod `stretch_damping
  0.05→1.25e3`, `bend_damping 0.02→0.3`.
- Soft-grid `k_damp 1.0→1e4` — NOT a rescale (the tet-damping *formulation*
  changed). Re-tuned empirically: a sweep showed the cable dent is
  stiffness-limited (~`k_damp`-insensitive) and `k_damp` only sets whether the
  cable *tumbles* the block; `1e4` is the lowest value that stays clean.
- **Global/solver-wide** damping (`soft_contact_kd`, `shape_material_kd` blanket
  fill) **reverted to Newton's native `1e-4`/`1e2`** per the user — these are
  model-global (affect every object's contacts), so left native, not inflated.

Net change — soft block **softened 4×** (`k_mu`/`k_lambda` ÷4) in ALL soft demos:
- `cable_soft` & `soft_compression`: 1e4/5e4 → **2.5e3/1.25e4**
- `soft_pickplace`: 2e3/1e4 → **5e2/2.5e3**
- `rigidCube_soft`: 5e2/2.5e3 → **1.25e2/6.25e2**

Verified (cuda:0, 720 frames, all `test_final` PASS, no NaN / fall-through):
- `cable_soft`: cable dent **6.8 → 14.1 mm** (softening ~doubled it), block stays
  (6 mm drift, 13° tilt). Clear win — reverting the global damping did NOT
  destabilize it.
- `rigidCube_soft`: **~0 mm compression** — the cube lands on the block *edge*
  (half-offset drop) and rolls straight off the now-soft edge without dwelling;
  softening can't help that scenario (needs a centered drop, or the force-limited
  gripper *placing* it — the other dev's territory).
- `soft_compression`, `soft_pickplace`: `test_final` PASS; deformation not
  directly measured (no `--npz` wrapper).

Cleanup / still pending:
- Deleted `_external/newton_core/` (6-file old-Newton snapshot used only for the
  diff; verified unused, off the import path, gitignored; recoverable from
  `_external/newton` git history).
- The repo **still depends on `_external/newton`** (editable install at the
  drifted commit). Decoupling — pip-pin upstream `newton` so `_external` is
  optional (planned "S0") — is NOT done.
- The other soft examples' *damping* is still un-migrated (native/under-damped);
  only their block *stiffness* was softened. `rigidCube_soft`/`soft_pickplace`
  carry the other dev's force-limited-gripper work — coordinate before editing.

## rigidCube_soft RoboLab variant

- New `examples/example_rigidCube_soft_franka_robolab.py`: RoboLab-graphics
  variant of `rigidCube_soft_franka` (heavy ~1 kg cube dropped on a pillow-soft
  block — compression is obvious). Subclasses `rigidCube_soft_franka.Example` and
  swaps only the viz layer, mirroring the cable variant; auto-discovered as
  `rigidCube_soft_franka_robolab` (no `__init__` edit). Verified: cube drops onto
  the block edge and rolls off; `test_final` passes; `combined.mp4` + frames
  produced. (Edge-drop-and-roll-off shows little block compression — see the
  damping/softening section above.)

## objectview inspection camera (both robolab examples)

- `--objectview True` adds a fixed `object_view_camera` framed on the soft body +
  surrounding table (posed from `soft_start_pos` via `look_at_quat_wxyz`,
  `focal_length=6.0` for a tight crop). It is rendered and dumped as **PNG stills
  only** — kept out of `combined.mp4`, so the kept video is unchanged.
- Mechanism: new `CameraConfig.in_combined_video` flag (`robolab_viz/config.py`);
  `robolab_viz/raycast.py` builds the combined video from only cameras with
  `in_combined_video=True` (per-camera PNG dumping is unchanged).
- `--png-every` default 60→30 in both robolab examples.
- Verified on both: objectview PNGs every 30 frames, `combined.mp4` stays
  1280×360 (two cameras, objectview excluded).

## Force-limited gripper + YCB pick-place demos (2026-06-16)

Net file changes this session:

- `examples/rigidCube_soft_franka.py` — **force-limited gripper** added
  (`--grip-force-control`, default on; `--grip-threshold`, default 15 N).
- NEW `examples/soft_pickplace_franka.py` — soft-body pick-and-place.
- NEW `examples/pickplace_ycb_franka.py` — rubik's-cube + banana + bowl friction/
  impact demo (RoboLab `RubiksCubeAndBananaTask` scene).
- NEW `examples/example_pickplace_ycb_robolab.py` — RoboLab-rendered variant →
  `outputs/.../robolab_preview/combined.mp4`.
- `examples/__init__.py` — registered `soft_pickplace_franka`, `pickplace_ycb_franka`.
- `CLAUDE.md` — added **Force-Limited Gripper** + **Obstacle (table) non-penetration**
  sections; the two new pick-place examples listed.

Force-limited gripper (replaces the commanded-width hack): the gripper is
position-controlled and creeps closed; a per-substep device latch
(`_update_grip_target_kernel`) reads the object→proxy contact reaction and, the
moment it crosses a threshold, latches the width and holds. No in-loop force
feedback → cannot eject. Rigid object → stops at the surface instantly; soft
object → compresses gradually until the threshold. Reaction source: rigid =
PUBLIC `SolverVBD.collect_rigid_contact_forces`; soft = recomputed from PUBLIC
`soft_contact_*` geometry (`ke·penetration`). Public Newton API only; nothing in
`_external` is modified or imported from `newton._src` (verified each session).

(Aside: whole-arm two-way coupling — feeding the grasp reaction onto the wrist so
the arm feels payload — was built and then reverted at the user's request. The
arm has the payload capacity to carry task objects; only the **gripper DOF**
should respond to contact, which the force-limited latch does.)

`pickplace_ycb_franka` specifics — FR3 at the world origin with the objects at the
demo's recorded poses; the recorded Panda+Robotiq joint trajectory does NOT
transfer (different mount/gripper/kinematics, EE lands ~15 cm off, below the
table), so we reuse the recorded grasp/release LOCATIONS and re-plan with IK. Cube
rendered as a 3×3-colored rubik's cube (54 sticker boxes on a black body, opposite
faces white/yellow, red/orange, blue/green). Cube dropped off-center on the bowl
rim → knocks the **dynamic** bowl ~23 cm (off-center-impact physics demo). Banana
grasped ~a quarter down (descends to the table), lifts, then slips — physical.

Mesh-object gotchas discovered (all in `pickplace_ycb_franka`):

- **Shared `Mesh` BVH segfault.** Re-using one `Mesh` object across the object
  model AND the combined viz model crashes: finalizing the viz model rebuilds the
  shared mesh's BVH and frees the GPU memory the object collision pipeline points
  at → segfault in `narrow_phase.launch_custom_write`. Fix: finalize the viz model
  BEFORE the object model (viz is render-only, so its now-stale BVH is harmless;
  the object model, finalized last, owns the live BVH).
- **Mesh midphase needs low-poly.** A full-res 16k-tri mesh BVH crashes the mesh
  midphase in a multi-shape scene; decimate to ~1.2k tris (`trimesh`
  quadric). Note: once ANY mesh collides, the midphase processes EVERY mesh shape
  in the model, including visual-only ones — keep them all low-poly.
- **Concave dynamic mesh needs a big contact buffer.** A box settling into the
  concave bowl generates thousands of triangle contacts; the default
  `rigid_body_contact_buffer_size=2048` overflows → penalty-force spike → both
  bodies ejected to infinity. Raised to 16384.
- **A *dynamic* concave mesh must be a CONVEX DECOMPOSITION, not the raw mesh.**
  Even with the big buffer, a box wedging into the raw concave bowl gets
  contradictory contact normals → a single-substep penalty spike (~1500 m/s) that
  ejects the WHOLE solve (cube/banana/bowl all to infinity; the banana looked like
  it "straightened and flew" but it was just collateral to the global blow-up).
  More substeps did NOT help (it is geometric, not dt). Fix: `_convex_pieces()`
  runs **coacd** convex decomposition (coacd is installed) → ~24 convex hull pieces
  as the bowl's collision shapes (the real mesh stays the visual). Convex hulls
  have consistent normals → stable, and preserve the cavity (generalizes to any
  concave object). IMPORTANT: do **not** quadric-decimate the hull pieces —
  decimation breaks convexity and the spike returns; use each part's exact hull.
- **Use realistic masses.** Naive `density` gave a ~23 g bowl / 46 g banana, so any
  contact flung them. Set densities to the YCB catalog masses (cube 0.2, banana
  0.12, bowl 0.5 kg) — both accurate and far more stable. Optimize for physical
  accuracy, not for how far the impact visibly moves the bowl.
- **Friction is real, don't clobber it.** Object↔gripper-pad grip is a VBD Coulomb
  body-body contact (tangential force bounded by `mu·normal`). Do NOT
  `object_model.shape_material_mu.fill_()` after finalize — it overwrites the
  per-shape `mu` and weakens the grip. Banana/pads set to `mu=2.0`.

## Light-body fling fix + solver-framework rule (2026-06-16, later)

- **CLAUDE.md**: added **Solver Framework Selection** (split MuJoCo+VBD whenever any
  deformable/soft is present; single `SolverMuJoCo` for rigid-only) and **Light-body
  contact stability** (the η diagnosis + the velocity-clamp fix). `pickplace_ycb_franka`
  is rigid-only but intentionally kept on VBD as the "VBD can host rigid meshes" proof.
- **Diagnosed** the `--bowl-mass 0.05` fling: VBD penalty/ALM contact force is
  mass-independent, so pair stability is `η = ke·dt²/m_reduced`; the blanket `ke=5e4`
  puts a 0.05 kg bowl at `η≈1.1` (pair with the cube `η≈1.36`) and the `alpha=0`
  correction ejects the lighter-coupled body. Confirmed chaotic: the **cube** (0.2 kg)
  flies to z≈−11 m / (42,0,29) m and the bowl to y≈−21 m, run-to-run varying — matches
  "the cube also flies away." Heavy bowl (0.5 kg, `η≈0.11`) is clean.
- **Fix** (`examples/pickplace_ycb_franka.py`): per-substep `_clamp_body_velocity_kernel`
  on the object `body_qd` (linear ≤ `--max-body-speed` 3.0 m/s, angular ≤ 50 rad/s),
  skipping kinematic proxies, launched after each VBD substep (inside the CUDA graph).
  Rigid analog of `particle_max_velocity`; public API only. Also wired `--bowl-mass`
  into the parser (was a bare `getattr` default).
- **Verified** (cuda:0, A100): at 0.05 kg the cube/bowl now peak 1.09/0.79 m/s and settle
  on the table — identical to the 0.5 kg baseline; peaks stay *below* the 3.0 clamp
  (escalation never starts). CUDA-graph capture OK; full-demo `--test` checked at both
  masses. Clamp is invisible in normal motion (legit max ~1.1 m/s).
