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
   `GraspWindow` with a cloth-scale `force_target` INSIDE the shell's achievable squeeze (~2 N —
   the target-relative admittance law then converges to a stable ~8–9 mm pinch and stays live to
   re-tighten; see [gripper.md](gripper.md)). Set
   `coupling_soft_ke=CFG.soft_contact_ke` to enable the grip-signal harvest — the value is just the
   opt-in; the framework centrally derives the effective ke as the HARMONIC mean of the cloth's and
   the pad's own authored ke (central compensation, solver-architecture.md "Contact materials"),
   the SAME derivation for every particle object — and
   `object_pipeline_kwargs={"soft_contact_margin": CFG.contact_margin}`; buffer sizes go in
   `object_solver_kwargs`.
5. **Timestep**: `substeps=10, vbd_iterations=5` (Newton's). dt is part of the contact parity — the
   config's dimensionless stiffness η = ke_eff·dt²/m matches Newton's at 10 substeps @ 60 fps.
6. **That's it for solver physics** — the framework auto-detects cloth (a shell: surface tris/edges,
   no tets) from the finalized model and applies `cloth_particle_kwargs(CFG)` (self-contact + radii)
   AND the central particle-contact material coupling (shape-side ke/kd DERIVED from each shape's
   own authored material so the contact lands on the harmonic mean; friction is fully per-object —
   no stamp, no exception) centrally. It also auto-routes to the split MuJoCo-robot + VBD-cloth
   path with gripper proxies.

### Bags use the same shell path

Deformable bags are registered with physics `kind: "cloth"` and semantic `category: "bag"`.
Their USDA meshes and reproducible Objaverse conversion manifest live under
`assets/objects/objaverse_bags/`; see its README for UIDs, CC-BY attribution, remeshing, and the
explicitly inferred material values. `rest_on_z=True` is the only placement extension: it anchors
a bag shell's lowest vertex at `center_pos.z`, whereas garments retain centroid placement.

The current generated-scene limit is **one total bag/cloth/squishy item**, enforced by
`scene_generator.validate_scene`. This is deliberately conservative until the framework is tested
with multiple particle deformables/material configurations in one object model.

## The config (`ClothConfig` in `assets.py`) — SI-converted Newton cloth

| field | default | Newton (cm-g) | conversion |
|---|---|---|---|
| `scale` | `0.01` | — | native USD units → m (shirt ≈ 0.65 m) |
| `flatten_z` | `0.12` | — | demos pass `1.0` (drop the inflated shirt, Newton's init) |
| `rest_on_z` | `False` | — | `True` anchors a bag's lowest vertex at its placement z; garments remain centroid-anchored |
| `density` | `0.2` kg/m² | `0.02` g/cm² | ×10 (≈ 0.17 kg shirt) |
| `particle_radius` | `0.008` | `0.8` cm | ×0.01 — the cloth↔body contact THICKNESS |
| `contact_margin` | `0.008` | `0.8` cm | ×0.01 (pipeline `soft_contact_margin`) |
| `tri_ke`/`tri_ka` | `10` | `1e4` | ×1e-3 ([M/T²]) |
| `tri_kd` | `1.5e-5` | `1.5e-2` | ×1e-3 |
| `edge_ke` | `5e-5` | `5` | ×1e-5 ([M·L/T²]: VBD bending force ≈ edge_ke·θ) |
| `edge_kd` | `5e-6` | `0.5` | ×1e-5 |
| `soft_contact_ke`/`kd`/`kf` | `15` / `0.015` / `1.0` | `1e4`/`1e1`/`1e3` | ×1e-3, then re-authored ×1.5 so the central harmonic coupling lands on Newton's proven EFFECTIVE 30 N/m / 0.03 (measured requirement; see below) |
| `soft_contact_mu` | `0.25` | `0.25` | dimensionless |
| `self_contact_radius`/`margin` | `0.002`/`0.002` | `0.2`/`0.2` cm | ×0.01; the radius sets the settled layer spacing (~2 mm) |
| `self_contact_rest_exclusion_radius` | `0.005` | `0.5` cm | ×0.01 |

### The unit-conversion law (THE cloth gotcha — how the grasp was broken for weeks)

Newton's cloth example runs in **centimetre-gram** units. Copying its numbers verbatim into this
metre-kilogram framework (the original `ClothConfig`) silently produced a cloth whose **contact was
~1000× stiffer and ~100× more damped relative to particle weight** than Newton's working grasp, on a
shirt **10× too light** (density "0.02" read as kg/m²). Individually each number looked plausible;
dimensionlessly the grasp physics was 3–4 orders of magnitude off:

- VBD body↔particle contact uses the **arithmetic mean** of `model.soft_contact_*` (particle side)
  and the touching **shape's stored material** (`mu`: geometric mean). Newton's example authors a
  shape profile giving pad↔cloth ke_eff = avg(10, 50) = **30 N/m**, kd_eff = **0.03** ≈ 0.5×
  critical, μ_eff = √(0.25·1.5) ≈ 0.61. This repo instead DERIVES the shape side centrally
  (harmonic coupling with the pads'/table's own authored ke=5e4 — solver-architecture.md "Contact
  materials") and authors the cloth's own ke/kd = **15 / 0.015** so the derivation lands on the
  SAME proven effective values: ke_eff = harm(15, 5e4) = **30 N/m** (η = 3.2), kd_eff = **0.03** =
  0.5× critical. The effective contact is the tuned quantity — at authored 10/0.01 (eff 20, η 2.1)
  the marginal pinch sheds at lift, measured 2026-07-10. The original verbatim cm-gram copy gave
  ke_eff 3e4 N/m (η ≈ 12 000) and kd_eff 55 (≈ 100× critical).
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

1. **Descend so the fingertip is COMMANDED 5 mm BELOW the table top** (`SURF = table_top − 0.005`,
   exactly Newton's recipe): the robot-side stopper holds the fingers at the surface and the excess
   becomes pressing normal force. This is what makes the grasp work at PHYSICAL per-object friction
   (cloth↔table eff √(0.25·0.5) ≈ 0.35): anchoring is μ·N, so N compensates μ. Measured 2026-07-11:
   without the press the pinch sheds at the drag onset even at eff 0.45 (fold fails, x-extent
   0.585); pressed, the fold is robust at the TRUE eff 0.35 (x-extent 0.454, engaged 615 frames,
   94 captured contacts — the plow actually gathers cloth into the jaw MORE easily on the
   slipperier table). Hovering at the cloth surface leaves the stack below the pinch plane.
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

Finger drive: `cloth_franka` uses the force `GripController` with `force_target=2 N` — INSIDE the
shell's achievable ~0.3–3 N squeeze, so under the target-relative admittance law
(`GripConfig.window_params`: gain ∝ 1/target, deadband ∝ target) the regulator latches at the 0.3 N
engage floor, tightens briskly, and CONVERGES to a true force equilibrium near the proven ~8–9 mm
jaw — staying live to re-tighten a shedding pinch ([gripper.md](gripper.md)).

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
The framework also applies the **central particle-contact material coupling** (every stored shape
ke/kd in the band re-targeted so Newton's arithmetic body↔particle mean lands on the HARMONIC mean
of the two objects' own authored values — a raw GRIP/FEM-scale ke≈5e4 would dominate the cloth
contact 1000:1; friction is fully per-object, no stamp) and overrides `coupling_soft_ke` to that
same effective ke so
the harvest (`grip.py`, f = ke_eff·pen) matches the true contact. A demo's
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
4. **The shape side of the cloth contact is DERIVED, never hand-authored.** VBD averages the
   shape's stored material into the contact, and one raw FEM-scale shape (ke 5e4) re-breaks the
   grasp — the framework therefore re-targets EVERY in-band shape's stored ke/kd in a particle
   scene onto the harmonic mean of the two objects' own authored values. A mixed cloth+rigid scene
   is covered automatically for the cloth contact; note the compensated stored values also soften
   that scene's rigid↔rigid pairs (solver-architecture.md "Contact materials").
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
[NVIDIA_cloth_manip.md](../external/NVIDIA_cloth_manip.md) (NVIDIA's recipe),
[solver-architecture.md](solver-architecture.md) ("Known limitations", standing open items).
