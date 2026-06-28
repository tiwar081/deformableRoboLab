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
- Cable shape contact material (`ke=2e4, mu=1.5`, **`kd=1e2` absolute**) is restored after the
  blanket object material fill. The `kd` was re-derived from the old `4e5` (≈1e4× critical, which
  dominated the grip with a spurious velocity-proportional force once the alpha=0 runaway was
  removed) — see CLAUDE.md and ONGOING.md.
- Timing: descend 0–2.8 s, close 2.8–4.0, hold 4.0–4.8, lift 4.8–6.8, sweep from 6.8 s at
  0.18 Hz, amplitude smoothstep-ramped over 1.5 s (a step in target velocity kicks the
  pinched cable out of the grasp).

## Soft body (FEM block)

The FEM grid from Newton's `rigid_soft_contact` example (the only upstream two-way VBD
rigid+soft scene), scaled to the table:

- `add_soft_grid(...)`, 4×4×4 cells of 0.0125 m (5×5×5 cm, 125 particles), centered at
  `soft_start_pos`. `density=150`, `k_damp=10` (absolute VBD damping units — see the
  Newton-version gotcha in CLAUDE.md; re-derive on every Newton bump).
- **One shared stiffness** `k_mu=5e2, k_lambda=2.5e3` (medium): soft enough to visibly dent/squash
  under the dropped cube and the pressing plate, firm enough to grasp and lift in `soft_pickplace`
  without squishing out of the pads. This single profile is a deliberate cross-comparability
  tradeoff over the old per-demo tuning (pillow-soft / firm / small-firm).
- Contact: `soft_contact_ke=1e5, kd=1e-4, kf=1e3, mu=0.8`, `particle_enable_tile_solve=False`,
  particle radius 0.0035 (the contact boundary sits one particle radius above the rendered surface —
  large radii read as contact-before-touching). The block tolerates the near-zero `kd` because its
  tet network + internal `k_damp=10` dissipate contact energy and it's grasped gently; a thin **cloth**
  shell does NOT (it needs ≈critical `soft_contact_kd` — see [cloths.md](cloths.md)).
  Note: `particle_max_velocity` is **not** set — it is inert under `SolverVBD` (XPBD/MPM only).
- The body-particle pair material is the **average** of `soft_contact_*` and the rigid shape's
  material, and VBD sums per-contact forces on the body. The pressing cube/plate keep their authored
  `ke=5e4`/`1e5` (registered by their asset builder, auto-restored after the blanket fill); the
  rigid↔particle dent is dominated by `soft_contact_ke=1e5` regardless.
- `rigid_body_particle_contact_buffer_size`: 4096 in the drop examples (a flat face contacts
  hundreds of particles; overflow drops contacts frame-to-frame → NaN), 512 in `cable_soft`.
- Two-way coupling stability ~ `sqrt(pair_ke / m_body) · substep_dt` and particle mass; the
  16-substep examples sit comfortably within it (upstream uses 32 substeps for a heavier impactor).

## Cloth (thin shell) — see [cloths.md](cloths.md)

Cloth (shirt/towel/sheet) is implemented as a separate deformable type: VBD `add_cloth_mesh`, its own
`ClothConfig` + `cloth_solver_kwargs` (particle self-contact ON, which the block omits), and a
≈critically-damped `soft_contact_kd` the light shell requires. The reference demo is `cloth_franka`.
Full authoring guide + gotchas: **[cloths.md](cloths.md)**.

## Future deformables (project goal)

Remaining target types: **zip-ties** (a stiff cable variant). Same dynamic-proxy two-way bridge +
light-element `η` stability constraints, and the same `_sync_viz_state` particle-copy rule. Start
from NVIDIA's cloth-manipulation recipe — [NVIDIA_cloth_manip.md](NVIDIA_cloth_manip.md).
