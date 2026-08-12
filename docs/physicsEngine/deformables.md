# Deformables (cable, soft FEM block)

These are the tuned values + the reasons behind them — re-derive if Newton is re-pinned
(see the Newton-version gotcha in CLAUDE.md). **All values now live in
`deformableManipulationTools/params.py`** (`CABLE`, `SOFT_BLOCK`) — one source of truth shared by
every example; the cable and FEM block are built by `deformableManipulationTools.assets.add_cable`
/ `add_soft_block`. Edit the params/builders, not the examples. There is now **one** `SOFT_BLOCK`
(the per-demo `SOFT_BLOCK_PILLOW/COMPRESS/PICK` variants were collapsed into it) so all four soft
demos are cross-comparable.

## Cable (VBD rod)

- `add_rod(..., wrap_in_articulation=True)`: radius 0.008, segment length 0.035, 15 nodes,
  friction 1.5, **density 1200** (realistic jacketed cable; lighter cables turn pinch-contact
  residuals into ejection kicks — see the `η` light-body note in solver-architecture.md).
- Laid with a **2 cm bow** (the cable layout in the example's `plan`): a perfectly straight round rod on a
  flat table has a free rolling mode (VBD has no rolling friction) and rolls off; the bow
  locks it geometrically. The grasp/IK target is the midpoint of nodes 3–4 of the bowed layout.
- The start-position clamp accounts for the full cable extent so the whole cable rests on the table.
- Cable shape contact material (`ke=2e4, mu=0.7`, **`kd=30` absolute**) is restored after the
  blanket object material fill. mu=0.7 is a realistic rubber/PVC jacket (was 1.5, a grasp cheat:
  cage eff vs pads is now √(0.7·0.8)=0.75 — the cage holds geometrically, verified). kd=30 is the
  2026-07-10 critical-damping re-derivation: pair eff vs pads ≈ 30 ≈ 0.9× critical for a 8.4 g
  node (the older 1e2 was ~3× critical; the original 4e5 was ≈1e4× critical and dominated the grip
  with a spurious velocity-proportional force once the alpha=0 runaway was removed).
- Timing: descend 0–2.8 s, close 2.8–4.0, hold 4.0–4.8, lift 4.8–6.8, sweep from 6.8 s at
  0.18 Hz, amplitude smoothstep-ramped over 1.5 s (a step in target velocity kicks the
  pinched cable out of the grasp).

## Soft body (FEM block) — a raspberry-like object

The FEM grid from Newton's `rigid_soft_contact` example (the only upstream two-way VBD
rigid+soft scene), re-authored 2026-07-10 to REPRESENT delicate fruit (raspberry-like):

- `add_soft_grid(...)`, 4×4×4 cells of 0.0125 m (5×5×5 cm, 125 particles), centered at
  `soft_start_pos`. `density=650` — the EFFECTIVE whole-berry density of a hollow drupelet
  aggregate (berries float; 81 g / 0.80 N for the cube). `k_damp=40` (absolute VBD damping
  units — see the Newton-version gotcha in CLAUDE.md; re-derive on every Newton bump).
- **One shared stiffness** `k_mu=2e3, k_lambda=1e4`: tissue E ≈ 5.7 kPa, ν ≈ 0.42 (soft-berry
  range) — sags ~8% under its own weight, visibly dents under the cube/plate, bruise-scale forces
  are a few N (crush literature: 5–10 N). One profile shared by every soft demo (cross-comparable).
- Contact: `soft_contact_ke=2.5e4, kd=13, kf=2.5e2, mu=0.5`, `particle_enable_tile_solve=False`,
  particle radius 0.0035 (the contact boundary sits one particle radius above the rendered surface —
  large radii read as contact-before-touching). The contact is the fruit's SKIN penalty, authored
  at rigid-comparable scale ON PURPOSE: the compliance lives in the tissue (a cell column is
  E·cell ≈ 71 N/m, ~500× softer than the contact), and rigid-comparable scale keeps every shape
  OUT of the harmonic band in FEM scenes so rigid↔rigid pairs (the cable cage) stay authored.
  kd=13 is a critical-damping derivation (pair eff vs pads ≈ 21 ≈ 1.9× critical — fruit lands
  dead, no viscous-glue scale). mu=0.5 is waxy berry skin (eff vs pads √(0.5·0.8) ≈ 0.63).
  A thin **cloth** shell is the opposite regime — no volume compliance, band-compensated soft
  effective contact (ke_eff 30 N/m, kd_eff 0.5× critical); see [cloths.md](cloths.md).
  Note: `particle_max_velocity` is **not** set — it is inert under `SolverVBD` (XPBD/MPM only).
- The body-particle pair material mixes `soft_contact_*` with the rigid shape's **stored**
  material (Newton: arithmetic ke/kd, geometric mu), and VBD sums per-contact forces on the body.
  The framework re-targets stored shape ke/kd only where the shape is decisively stiffer than the
  particle side (k > 2.5·s — nothing, in FEM scenes, by construction; see
  solver-architecture.md "Contact materials"). The pressing cube/plate keep their authored
  `ke=5e4`/`1e5` (registered by their asset builder, auto-restored after the blanket fill); the
  rigid↔particle dent is dominated by the TISSUE regardless.
- Delicate-grasp physics (measured, `soft_pickplace`): the tissue's quasi-static pushback through
  the jaw is ~54 N/m secant — squeeze builds SLOWLY (the demo dwells ~4.4 s before lifting and
  lifts slowly), and sub-2 N targets plateau near ~1 N of squeeze, which lifts the berry but sheds
  it at the first lateral acceleration. `force_target=3.5 N` carries pick→place with the peak
  force at ~3.2 N — well under the 5 N crush threshold.
- `rigid_body_particle_contact_buffer_size`: 4096 in the drop examples (a flat face contacts
  hundreds of particles; overflow drops contacts frame-to-frame → NaN), 512 in `cable_soft`.
- Two-way coupling stability ~ `sqrt(pair_ke / m_body) · substep_dt` and particle mass; the
  16-substep examples sit comfortably within it (upstream uses 32 substeps for a heavier impactor).

## Imported squishies (`soft_mesh` — a `.tet` FEM object from a scan)

`SoftMeshConfig` + `assets.add_soft_mesh_object` mirror `SOFT_BLOCK`'s material/contact schema
exactly, so the centralized soft-contact coupling, proxy harvest, and FEM particle-solver config
apply unchanged — only the geometry source differs (imported tets instead of a procedural grid).
Catalog entries (`sponge`, `foam_brick`, `banana_soft`) override density/`k_mu`/`k_lambda` from the
object's real material class; every derived value is recorded in that entry's `inferred` list.

Two findings that cost real time — **do not re-walk them**:

- **Tetrahedralize with fTetWild, never tetgen.** tetgen surface-constrained tets carry sliver
  elements (min |vol| 3e-16…1e-10 m³ — *worse* with quality switches) whose ill-conditioning makes
  the VBD FEM solve gain energy at rest and NaN within seconds, **at any contact stiffness**.
  fTetWild (DefGraspSim's own tool, wrapped by `assets/objects/_utils/make_tet.py` via the
  `wildmeshing` bindings — an asset-prep dependency only) gives well-conditioned coarse tets: all
  three squishies then settle DEAD still (max |qd| = 0.000).
- **An imported mesh must re-derive its contact skin at its OWN per-particle mass.** The contact
  `soft_contact_*` set is not portable between meshes: `SOFT_BLOCK`'s raspberry-scale skin dropped
  onto a mesh with ~50× lighter particles measured **~90× critical** pair damping and NaN'd. The law
  the catalog entries follow: `ke = 2.5e4·(m_p / 9e-4 kg)` (holds the pair at the `SOFT_BLOCK` η
  operating point), `kd` from pair critical damping (~1.5× crit), `kf = 0.01·ke`. This is the same
  class of error as copying Newton's cm-gram numbers (CLAUDE.md): each value looks plausible alone,
  and only the dimensionless group relative to particle mass reveals it.

## Cloth (thin shell) — see [cloths.md](cloths.md)

Cloth (shirt/towel/sheet) is implemented as a separate deformable type: VBD `add_cloth_mesh`, its own
`ClothConfig` (Newton's `example_cloth_franka` cloth converted unit-consistently to SI — never copy its
cm-gram numbers verbatim) + centralized `cloth_particle_kwargs` (particle self-contact ON, which the
block omits) + a framework-applied shape material profile for everything the cloth touches. The
reference demo is `cloth_franka` (full grasp-and-fold). Full guide + gotchas: **[cloths.md](cloths.md)**.

## Future deformables (project goal)

Remaining target types: **zip-ties** (a stiff cable variant). Same dynamic-proxy two-way bridge +
light-element `η` stability constraints, and the same `_sync_viz_state` particle-copy rule. Start
from NVIDIA's cloth-manipulation recipe — [NVIDIA_cloth_manip.md](../external/NVIDIA_cloth_manip.md).
