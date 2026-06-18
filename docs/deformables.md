# Deformables (cable, soft FEM block)

These are the tuned values + the reasons behind them — re-derive if Newton is re-pinned
(see the Newton-version gotcha in CLAUDE.md). **All values now live in `assets/params.py`**
(`CABLE`, `SOFT_BLOCK`, `SOFT_BLOCK_PILLOW`, `SOFT_BLOCK_COMPRESS`, `SOFT_BLOCK_PICK`) — one
source of truth shared by every example; edit there, not in the examples.

## Cable (VBD rod)

- `add_rod(..., wrap_in_articulation=True)`: radius 0.008, segment length 0.035, 15 nodes,
  friction 1.5, **density 1200** (realistic jacketed cable; lighter cables turn pinch-contact
  residuals into ejection kicks — see the `η` light-body note in solver-architecture.md).
- Laid with a **2 cm bow** (`_cable_layout_positions`): a perfectly straight round rod on a
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

- `add_soft_grid(...)`, 4×4×4 cells of 0.0125 m (5×5×5 cm, 125 particles, ~12.5 g),
  centered at `soft_start_pos`. `density=100`, `k_damp` re-tuned per Newton 1.4 (see gotcha).
- Stiffness per example (softened 4× for visible deformation under Newton 1.4): `cable_soft`
  & `soft_compression` `k_mu=2.5e3,k_lambda=1.25e4`; `soft_pickplace` `5e2/2.5e3`;
  `rigidCube_soft` pillow-soft `1.25e2/6.25e2`.
- Contact (upstream values): `soft_contact_ke=1e5, kd=1e-4, kf=1e3, mu=0.3`,
  `particle_max_velocity=50`, `particle_enable_tile_solve=False`, particle radius 0.0035
  (the contact boundary sits one particle radius above the rendered surface — large radii
  read as contact-before-touching).
- The body-particle pair material is the **average** of `soft_contact_*` and the rigid
  shape's material, and VBD sums per-contact forces on the body. Dropping cube/sheet shapes
  are restored to `ke=1e5,kd=1e-4` so their pair matches upstream's sphere-grid pairing;
  averaging against the stiff table/pad shapes keeps body-body contacts stiff regardless.
- `rigid_body_particle_contact_buffer_size`: 4096 in the drop examples (a flat face contacts
  hundreds of particles; overflow drops contacts frame-to-frame → NaN), 512 in `cable_soft`.
- Two-way coupling stability ~ `sqrt(pair_ke / m_body) · substep_dt` and particle mass; the
  16-substep examples sit comfortably within it (upstream uses 32 substeps for a heavier impactor).

## Future deformables (project goal)

Target object types beyond the rod/block: **zip-ties, clothing, towels** (cloth). Cloth in
Newton is also VBD (`add_cloth_grid`/mesh); expect the same dynamic-proxy two-way bridge +
light-element `η` stability constraints to dominate, and the same `_sync_viz_state`
particle-copy rule. A cloth grip would use particle-colliding proxies + the soft harvest, like
`soft_pickplace`.
