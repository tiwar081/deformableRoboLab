# Cloth deformables — how to add a shirt / towel / sheet

How to set up a **cloth-type** deformable (a thin 2-D shell: shirt, towel, napkin, flag, …) in
this sim so the robot can manipulate it. Cloth runs on `SolverVBD` like the cable and FEM block,
but a thin shell is the most numerically demanding deformable here, so it has its **own** centralized
config + solver kwargs and a few rules the volumetric block does not need. Everything physical lives
in the package (`deformableManipulationTools/`); a demo declares only the scene + policy.

The reference implementation is `cloth_franka` (a vendored T-shirt). Copy its shape, change the
config — do **not** put cloth physics inline in a new demo.

## TL;DR — adding a new cloth object

Demos are DATA FILES (`examples/<name>.py` → a `DemoSpec`; see [examples.md](examples.md)). Adding a
cloth needs only scene + policy — the cloth solver physics is centralized:

1. **Author/obtain a triangle-mesh USD** of the cloth surface (a thin shell, not a solid). Drop it
   under `assets/objects/` (e.g. `towel.usd`).
2. **Add a config**: make a new `ClothConfig(...)` instance (or reuse the default) in
   `deformableManipulationTools/assets.py` — set `usd_file`/`usd_prim`, `scale` (native units → m),
   `flatten_z` (squash a draped/worn mesh flat for a table start), and `density` [kg/m²]. Leave the
   stiffness / contact / self-contact fields at the defaults unless you have a reason.
3. **Declare it in the data file's `scene`**: `Obj("cloth", CFG, pos=(x, y, z))` (+ an `Obj("proxies")`).
4. **Grasp**: set `coupling_soft_ke=CFG.soft_contact_ke` (proxy↔particle harvest) on the `DemoSpec`, and
   on the grasp give a shell-scale `force_target` (~3–8 N — the 30 N default is a rigid-box value, wrong
   for fabric). `object_pipeline_kwargs={"soft_contact_margin": CFG.contact_margin}`; buffer sizes go in
   `object_solver_kwargs`.
5. **That's it for solver physics** — the framework auto-detects cloth (a shell: surface tris/edges, no
   tets) from the finalized model and applies `cloth_particle_kwargs(CFG)` (self-contact + the radii)
   centrally. You never wire `particle_enable_self_contact` etc. in the demo. It also auto-routes to the
   split MuJoCo-robot + VBD-cloth path with the gripper proxies, same as the FEM-block demos.

## The config (`ClothConfig` in `assets.py`)

| field | default | meaning / when to change |
|---|---|---|
| `usd_file` / `usd_prim` | `unisex_shirt.usd` | the vendored shell mesh under `assets/objects/` |
| `scale` | `0.01` | native USD units → metres. Check the result is the real-world size (a shirt ≈ 0.5–0.7 m). |
| `flatten_z` | `0.12` | squash a 3-D draped/worn mesh in z so it **starts laid flat** on the table (a thick draped start envelops the gripper proxies and drags the arm). A towel authored flat → set `1.0`. |
| `density` | `0.3` | per-**area** [kg/m²] (≈0.13 kg shirt). Towel ≈ 0.2–0.4. This sets the per-particle mass, which the contact damping is derived against (below). |
| `particle_radius` | `0.005` | contact thickness of the shell. |
| `tri_ke`/`tri_ka`/`tri_kd` | `1e4`/`1e4`/`1e-2` | in-plane stretch/shear. Stiff (cloth barely stretches). |
| `edge_ke`/`edge_kd` | `5.0`/`0.5` | bending. Low = drapes easily; raise for a stiffer fabric (canvas, denim). |
| `soft_contact_ke`/`kd` | `1e4`/`1e1` | **proxy↔cloth + table↔cloth penalty. THE critical pair — see gotchas.** |
| `soft_contact_kf`/`mu` | `1e3`/`0.8` | tangential contact stiffness / friction. |
| `self_contact*` | on, r=0.002, m=0.003, filter=1, rest-excl=0.006 | particle self-collision so folds/layers don't pass through each other (see gotchas). |

## Solver kwargs — centralized (the demo declares none)

Particle solver config is applied **centrally** by `framework._particle_solver_config`, which picks by
deformable type: an FEM block (has tets) → `PARTICLE_SOLVER_KWARGS` (self-contact **off** — a volume
can't fold through itself); a cloth (a shell: surface tris/edges, no tets) → `cloth_particle_kwargs(cfg)`
(self-contact **on** + the radii). The demo's `object_solver_kwargs` holds only the scene-specific
`rigid_body_*` buffer sizes and merges on top. So a cloth demo never hand-rolls solver physics — that's
the point of the centralization.

## Grasp — the one demo knob is `force_target`

The unified admittance `GripController` is identical for cloth, cable, rigid, and FEM block; the only
per-demo knob is `GraspWindow.force_target` [N]. Fabric is gentle — pick **~3–8 N** (the thin-shell /
cable regime). Lowering the target also lowers the engage threshold (`clamp(0.15·target, 0.3, 2.0)`),
so the grip latches on a lighter touch instead of over-closing. Nothing else about the grasp is
demo-tunable; fix anything else centrally.

## Gotchas (cloth-specific; ordered by how badly they bite)

1. **Contact damping is mandatory and must be ≈critical, NOT carried over from the soft block.**
   The #1 cloth failure: ultra-light shell particles (~3e-5 kg) have no volumetric tet network or
   internal damping to absorb a contact impulse (the FEM block does — and so *masks* an under-damped
   contact). An over-stiff/undamped pad contact **ejects the particles to NaN** the moment the pads
   touch. Use `kd ≈ kd_crit = 2·√(soft_contact_ke · m_particle)` (≈1–4 here) — `1e1` is safely
   over-critical and matches Newton's `example_cloth_franka`. The old `ke=1e5 / kd=1e-4` (the firm
   soft-block contact) was ~5 orders below critical and blew up. **Re-derive `kd` if you change
   `density`, `soft_contact_ke`, the particle count, or re-pin Newton.**
2. **Self-contact must be ON for any folding/draping.** Without it, layers (and the table) pass
   straight through each other. It's enabled centrally via `cloth_particle_kwargs`. The topological filter
   (`threshold=1`) + rest-shape exclusion radius stop mesh-adjacent vertices from self-colliding at
   rest — tune the exclusion radius up if a flat cloth jitters/puffs at start.
3. **`soft_contact_ke` is read in two places — keep them one source.** The grip-force *harvest*
   (`grip.py`) reconstructs the pad reaction as `ke·penetration` (Newton exposes no body→particle
   force readback). The demo wires `coupling_soft_ke = CFG.soft_contact_ke` and the framework sets
   `model.soft_contact_ke` from the same field, so they move together. Never hard-code a different ke
   anywhere, or the reported grip force desyncs from the real one.
4. **`particle_max_velocity` does nothing under VBD.** It's honored only by XPBD/MPM. Don't set it
   and don't rely on it as a safety net — stability comes from the contact material (gotcha 1).
5. **Copy `particle_q`/`particle_qd` to the viz state** (the framework's `_sync_viz_state` already
   does this when `has_particles=True`). Forgetting it shows the cloth frozen / penetrated. Set
   `has_particles = True` on the demo class.
6. **A flat sheet lying on the table can't be pinch-grasped by a top-down jaw — confirmed empirically.**
   Two failure layers: (a) the settled shell is thin (~6 mm proud of the table), so a grasp tuned for
   a thicker object closes ABOVE it on air ("nothing happens") — descend to the cloth SURFACE; and
   (b) even contacting, a vertical jaw closing horizontally presses the sheet DOWN into the table — the
   reaction is vertical and projects to ≈0 on the jaw (closing) axis, so the squeeze never builds and
   the grip can't latch. Instrumented sweeps confirm NO lift across grasp-height, ±tilt, a diagonal
   "scoop" approach, AND both robots (panda/fr3); the closing-axis squeeze never exceeds a ~4 N
   transient. Friction is NOT the cause (effective cloth↔pad μ = √(μ_particle·μ_pad) ≈ 0.89 here, ABOVE
   Newton's ≈0.61). This is a grasp-*strategy* limit, **not** a physics/contact/friction/robot bug.
   To actually LIFT cloth you need a SCOOP: present a graspable feature (drape a corner over an edge /
   backstop, or pre-lift a corner) and sweep a finger UNDER a free edge before closing — Newton's
   `example_cloth_franka` does exactly this with a ~45° tilt + a multi-waypoint corner grasp.
   `solve_gripper_ik(..., tilt=, tilt_axis=)` supports the tilted approach such a primitive needs.
   Leave the failure visible; don't fake the grasp.
7. **The DEEPER blocker is the gripper-proxy bridge itself (architectural).** `cloth_franka` now
   replicates Newton's exact scoop motion (45° tilt + the per-corner approach→descend-to-surface→
   close→lift→drag→release), yet still does not lift. Instrumented: the dynamic **proxy** (the box that
   mirrors each finger in the VBD world — see [gripper.md](gripper.md)) **jams the rigid table at
   ~177 N** instead of wedging UNDER the thin cloth edge, and a pressing+sliding pad does not carry the
   cloth (μ=0.25 vs μ=0 give the same ≈0 drag). Newton's robot fingers contact the cloth DIRECTLY in
   one VBD solver; our split MuJoCo-robot↔VBD-cloth coupling can't reproduce a delicate cloth scoop. A
   faithful cloth pickup likely needs a finger-shaped collider IN the cloth's VBD solver (not a proxy
   that must squeeze between cloth and table), or a draped/standing cloth presentation. This is the
   real frontier for cloth manipulation here — beyond friction, height, tilt, or motion replication.

## Scale note (why metre scale is fine here)

Newton's standalone cloth example runs in **centimetre** scale "for better VBD numerical behaviour."
This framework runs everything (robot, table, cable, block) in **metre** scale, and a properly
**damped** cloth contact (gotcha 1) is stable at metre scale too — verified: the shirt no longer
ejects or tunnels. So a cm-scale migration is **not** needed; if you ever hit a residual VBD
absolute-length-threshold issue (very fine folds), prefer adding substeps / VBD iterations before
considering a scale change, which would be invasive (the proxy coupling bridges the metre-scale
MuJoCo robot to the object world and would need unit conversion everywhere).

See also: [deformables.md](deformables.md) (cable + FEM block), [gripper.md](gripper.md) (the unified
grip + harvest), [solver-architecture.md](solver-architecture.md) (the split-solver routing),
[NVIDIA_cloth_manip.md](NVIDIA_cloth_manip.md) (NVIDIA's recipe).
