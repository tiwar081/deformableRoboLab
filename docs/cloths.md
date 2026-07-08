# Cloth deformables — how to add a shirt / towel / sheet, and how the cloth grasp works

How to set up a **cloth-type** deformable (a thin 2-D shell: shirt, towel, napkin, flag, …) in
this sim so the robot can manipulate it. Cloth runs on `SolverVBD` like the cable and FEM block,
but a thin shell is the most numerically demanding deformable here, so it has its **own** centralized
config + solver kwargs and a few rules the volumetric block does not need. Everything physical lives
in the package (`deformableManipulationTools/`); a demo declares only the scene + policy.

The canonical cloth is **Newton's `example_cloth_franka` shirt converted unit-consistently to SI**
(`ClothConfig` defaults). The demo is `cloth_franka`: a vendored T-shirt, dynamic-proxy grip, and one
force-grip hot-dog fold. Copy the demo's shape, change the config — do **not** put cloth physics
inline in a new demo.

## TL;DR — adding a new cloth object

Demos are DATA FILES (`examples/<name>.py` → a `DemoSpec`; see [examples.md](examples.md)). Adding a
cloth needs only scene + policy — the cloth solver physics is centralized:

1. **Author/obtain a triangle-mesh USD** of the cloth surface (a thin shell, not a solid). Drop it
   under `assets/objects/` (e.g. `towel.usd`).
2. **Add a config**: make a new `ClothConfig(...)` instance in `deformableManipulationTools/assets.py`
   — set `usd_file`/`usd_prim`, `scale` (native units → m), `flatten_z` (1.0 = drop as authored;
   <1 squashes a draped/worn mesh flat for a table start), and `density` [kg/m²] if the fabric
   differs. **Leave the stiffness / contact / self-contact fields at the defaults** — they are the
   SI-converted Newton cloth and are matched as a SET (see the conversion law below).
3. **Declare it in the data file's `scene`**: `Obj("cloth", CFG, pos=…)` + `Obj("proxies")`.
4. **Grasp** (see the recipe below): descend so the fingertip point reaches the table top; default
   proxy pad (the finger's true collider) on the proxy path. Fingers: EITHER an explicit
   `finger_schedule` closing to a FIXED ~8 mm total jaw gap (Newton's recipe, the proven path), OR a
   `GraspWindow` with a cloth-scale `force_target` ≤ 2 N (the force-grip trial — engage rides the
   0.3 N floor and the deadband holds the latched width; see [gripper.md](gripper.md)). Set
   `coupling_soft_ke=CFG.soft_contact_ke` to enable the grip-signal harvest (the framework
   replaces it with the correct shape-averaged value centrally) and
   `object_pipeline_kwargs={"soft_contact_margin": CFG.contact_margin}`; buffer sizes go in
   `object_solver_kwargs`.
5. **Timestep**: `substeps=10, vbd_iterations=5` (Newton's). dt is part of the contact parity — the
   config's dimensionless stiffness η = ke_eff·dt²/m matches Newton's at 10 substeps @ 60 fps.
6. **That's it for solver physics** — the framework auto-detects cloth (a shell: surface tris/edges,
   no tets) from the finalized model and applies `cloth_particle_kwargs(CFG)` (self-contact + radii)
   AND the cloth-scene shape material profile (pads + object-side table get `CFG.shape_contact_*`)
   centrally. It also auto-routes to the split MuJoCo-robot + VBD-cloth path with gripper proxies.

## The config (`ClothConfig` in `assets.py`) — SI-converted Newton cloth

| field | default | Newton (cm-g) | conversion |
|---|---|---|---|
| `scale` | `0.01` | — | native USD units → m (shirt ≈ 0.65 m) |
| `flatten_z` | `0.12` | — | demos pass `1.0` (drop the inflated shirt, Newton's init) |
| `density` | `0.2` kg/m² | `0.02` g/cm² | ×10 (≈ 0.17 kg shirt) |
| `particle_radius` | `0.008` | `0.8` cm | ×0.01 — the cloth↔body contact THICKNESS |
| `contact_margin` | `0.008` | `0.8` cm | ×0.01 (pipeline `soft_contact_margin`) |
| `tri_ke`/`tri_ka` | `10` | `1e4` | ×1e-3 ([M/T²]) |
| `tri_kd` | `1.5e-5` | `1.5e-2` | ×1e-3 |
| `edge_ke` | `5e-5` | `5` | ×1e-5 ([M·L/T²]: VBD bending force ≈ edge_ke·θ) |
| `edge_kd` | `5e-6` | `0.5` | ×1e-5 |
| `soft_contact_ke`/`kd`/`kf` | `10` / `0.01` / `1.0` | `1e4`/`1e1`/`1e3` | ×1e-3 |
| `soft_contact_mu` | `0.25` | `0.25` | dimensionless |
| `shape_contact_ke`/`kd`/`mu` | `50` / `0.05` / `1.5` | `5e4`/`5e1`/`1.5` | ×1e-3 — pads + table (framework-applied) |
| `self_contact_radius`/`margin` | `0.002`/`0.002` | `0.2`/`0.2` cm | ×0.01; the radius sets the settled layer spacing (~2 mm) |
| `self_contact_rest_exclusion_radius` | `0.005` | `0.5` cm | ×0.01 |

### The unit-conversion law (THE cloth gotcha — how the grasp was broken for weeks)

Newton's cloth example runs in **centimetre-gram** units. Copying its numbers verbatim into this
metre-kilogram framework (the original `ClothConfig`) silently produced a cloth whose **contact was
~1000× stiffer and ~100× more damped relative to particle weight** than Newton's working grasp, on a
shirt **10× too light** (density "0.02" read as kg/m²). Individually each number looked plausible;
dimensionlessly the grasp physics was 3–4 orders of magnitude off:

- VBD body↔particle contact uses the **arithmetic mean** of `model.soft_contact_*` (particle side)
  and the touching **shape's material** (`mu`: geometric mean). Newton-effective pad↔cloth:
  ke_eff = avg(10, 50) = **30 N/m**, kd_eff = **0.03** ≈ 0.5× critical, μ_eff = √(0.25·1.5) ≈ 0.61.
  The old values gave ke_eff 3e4 N/m (η = ke_eff·dt²/m ≈ 12 000 vs Newton's 3.2) and kd_eff 55
  (≈ 100× critical).
- Consequences, all measured on the failing demo: newton-scale grip forces became 50–80 N spikes
  that **expelled the shell from the pinch** (watermelon-seed ejection) and the over-critical kd
  acted as viscous glue that ratcheted particles out of the jaw during any motion.
- Conversion law (cm,g → m,kg): stiffness-like [M/T²] ×1e-3 · damping-like [M/T] ×1e-3 · bending
  [M·L/T²] ×1e-5 · area density ×10 · lengths ×0.01 · **and match dt** (10 substeps @ 60 fps).
  If you re-derive from a future Newton example, convert the whole SET together and check
  η = ke_eff·dt²/m and kd_eff/(2√(ke_eff·m)) against the source.

## The grasp recipe (Newton's, verified knob-by-knob on our stack)

Measured from Newton's recorded result (`_external/newton/cloth_franka.usd`) and reproduced here
with instrumented probes — all knobs matter; each was A/B-tested:

1. **Descend so the fingertip grasp point reaches the TABLE TOP** (`SURF = table_top + 0`), plowing
   the pad through the full cloth stack. (Newton even commands 5 mm below.) Hovering at the cloth
   surface leaves the stack below the pinch plane.
2. **Close to a FIXED ~8 mm total jaw gap** (`SHUT = 0.004` per finger) and **never 0**: the cloth
   between the pads buckles up into a multi-layer pucker (~20 mm effective thickness against the
   8 mm-radius contact), and the finite gap holds it in deep, stable penalty penetration. Closing to
   0 measurably expels the captured wad (17 particles at 6 mm gap → 0 at full close). The MuJoCo
   fingers feel no cloth (it exists only in VBD), so nothing else can stop them.
3. **Straight-down pinch** (tilt 0; Newton uses ≤ ~17°), grasp a region near an edge/corner, then
   lift and drag — the fold. Holding force is PHYSICAL friction (μ_eff ≈ 0.61): ~30–50 particles
   stay in the jaw through the whole fold at ~0.2–0.7 N per-pad reaction.
4. **Proxy path: the DEFAULT pad is the finger's own collider, deep-copied.** Mesh pads must own
   their BVH inside the object model, and the left/right finger proxies are collision-filtered.

Finger drive: `cloth_franka` TRIALS the force `GripController` via a
`GraspWindow` with `force_target=1 N`: any target ≤ 2 N keeps the engage threshold at its 0.3 N
floor, so the blind close stops at the shell's first 0.3 N of squeeze and the 2 N deadband then
freezes that width — effectively a force-triggered fixed-gap pinch. A thin shell's reaction is
newton-scale, so the rigid-scale constants remain a caveat ([gripper.md](gripper.md)).

### Both grip paths work — measured comparison (Newton-parity physics, single-fold isolation runs)

| path | mechanism | fold result |
|---|---|---|
| dynamic proxies, TWO-WAY (EE feedback on), default finger-collider pad (`cloth_franka`) | proxy bridge | **full fold**, cloth centroid +12.9 cm |

(See the `cloth_franka` docstring and [examples.md](examples.md).)

The old claim "the proxy bridge architecturally cannot grip cloth" is **disproven** — it was the
unit-blind parameters + the close-to-zero jaw. The EE feedback was also exonerated directly (a
one-way A/B of the proxy path behaves identically; the harvested wrench at correct parameters is
sub-newton). Note **Newton's own example is NOT a two-way single-solver grasp**: its Featherstone
robot runs with gravity zeroed and rigid contacts disabled (it feels neither cloth nor table) while
`SolverVBD(integrate_with_external_rigid_solver=True)` collides cloth against externally-posed
finger shapes: strictly one-way, fingers position-commanded to the fixed gap.

## Solver kwargs — centralized (the demo declares none)

Particle solver config is applied **centrally** by `framework._particle_solver_config`, which picks by
deformable type: an FEM block (has tets) → `PARTICLE_SOLVER_KWARGS` (self-contact **off**); a cloth
(a shell: surface tris/edges, no tets) → `cloth_particle_kwargs(cfg)` (self-contact **on** + radii).
The framework also applies the cloth-scene **shape material profile** (`shape_contact_*` on the
pads/fingers + object-side table, AFTER `restore_proxy_materials` — the GRIP/FEM-scale ke≈5e4 would
dominate the averaged cloth contact 1000:1) and overrides `coupling_soft_ke` to the effective
avg(soft, shape) ke so the harvest (`grip.py`, f = ke_eff·pen) matches the true contact. A demo's
`object_solver_kwargs` holds only the scene-specific `rigid_body_*` buffer sizes.

## Gotchas (cloth-specific; ordered by how badly they bite)

1. **Never copy Newton cm-gram numbers verbatim** — convert the whole parameter set with the law
   above and verify the dimensionless groups. This single mistake produced every historical cloth
   "impossibility" below.
2. **Never close the jaw to 0 on a shell.** A penalty pinch whose gap goes below the stack's minimum
   must expel the cloth — capture is a FINITE-gap deep-penetration state (8 mm for this shirt).
3. **Self-contact must be ON for any folding/draping** (centralized via `cloth_particle_kwargs`).
   Without it layers pass through each other. The 2 mm self-contact radius sets the settled
   two-layer spacing; there is no authored gap between the shirt's layers (one closed shell).
4. **Any shape the cloth touches needs `shape_contact_*`-scale material.** VBD averages the shape
   material into the contact; one FEM-scale shape (ke 5e4) re-breaks the grasp. The framework
   covers pads + table; a future mixed cloth+rigid scene must extend the profile to the new shapes.
5. **`particle_max_velocity` does nothing under VBD** — stability comes from the contact material.
6. **Copy `particle_q`/`particle_qd` to the viz state** (`_sync_viz_state` does this when
   `has_particles=True`) or the cloth renders frozen/penetrated.
7. **A flat settled shirt IS pinch-graspable** — the old "flat cloth can't be top-down grasped"
   conclusion was an artifact of gotchas 1+2 (and a too-shallow descent). With the recipe the pads
   plow to the table, the sheet buckles into the jaw, and the fixed gap holds it.
8. **Mesh finger proxies must be deep-copied.** A full-mesh proxy must DEEP-COPY the finger mesh
   (`build_gripper_proxies` does) — aliasing the robot's `Mesh` frees its BVH under the object narrow
   phase (error 700). Keep `rigid_body_particle_contact_buffer_size` large for mesh pads.

## Scale note (metre scale is fine — with converted parameters)

Newton's cloth example runs in centimetre scale "for better VBD numerical behaviour". This framework
runs everything in metre scale, and with the SI-converted parameter set above the same sim is stable
and grasps identically — verified end-to-end. The earlier instability history (ejections, NaNs) was
the unit mismatch, not the scale. If you ever hit a residual VBD absolute-length-threshold issue,
prefer more substeps/iterations over a scale migration (the proxy coupling bridges the metre-scale
MuJoCo robot and would need unit conversion everywhere).

See also: [deformables.md](deformables.md) (cable + FEM block), [gripper.md](gripper.md) (the unified
grip + harvest), [solver-architecture.md](solver-architecture.md) (the split-solver routing),
[NVIDIA_cloth_manip.md](NVIDIA_cloth_manip.md) (NVIDIA's recipe), [ONGOING.md](ONGOING.md) (open items).
