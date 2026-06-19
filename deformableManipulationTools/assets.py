"""Centralized object/asset builders. Every demo adds its scene objects through these so the
collision/representation NUANCES (mesh-vs-viz split, coacd convex decomposition for concave
meshes, FEM grids, contact materials, realistic masses) live in ONE place and a new demo cannot
re-introduce a per-object collision bug (e.g. the raw-concave-mesh ejection, docs/SOLVERS.md §4).

An example only chooses WHICH objects to add and WHERE (the scene); the physical properties come
from :mod:`deformableManipulationTools.params`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import warp as wp

import newton

from .params import (CABLE, PLATE, TABLE, CableConfig, PlateConfig, RigidBoxConfig, SoftBlockConfig,
                     TableConfig, YcbMeshConfig)
from . import mesh_collision as _mc

# Vendored scene meshes live under assets/objects/ (data, referenced by path; not imported).
OBJECTS_DIR = Path(__file__).resolve().parent.parent / "assets" / "objects"


def _register_material(builder, *, shapes=None, bodies=None, ke=None, kd=None, mu=None) -> None:
    """Record an object's AUTHORED contact material on the builder so the framework can restore it
    after its blanket proxy-fill (which clobbers all shape materials for a uniform grip contact).
    Centralizing this here means an example never has to re-apply a material override by hand."""
    store = getattr(builder, "_robolab_material_restores", None)
    if store is None:
        store = builder._robolab_material_restores = []
    entry = {"ke": ke, "kd": kd, "mu": mu}
    if shapes is not None:
        entry["shapes"] = [int(s) for s in shapes]
    if bodies is not None:
        entry["bodies"] = [int(b) for b in bodies]
    store.append(entry)

# Identical particle-contact SolverVBD config for every FEM-block demo (one source of truth).
PARTICLE_SOLVER_KWARGS = dict(
    particle_self_contact_radius=0.003, particle_self_contact_margin=0.005,
    particle_enable_self_contact=False, particle_enable_tile_solve=False,
    particle_vertex_contact_buffer_size=32, particle_edge_contact_buffer_size=64,
    particle_collision_detection_interval=-1,
)


def add_table(builder: newton.ModelBuilder, table: TableConfig = TABLE) -> int:
    """Visible object-side table collider (mirrors the hidden robot-side stopper). Returns the
    shape index (pass it to build_gripper_proxies so the proxies are filtered against it)."""
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, ke=table.object_ke, kd=table.object_kd, mu=table.object_mu)
    return builder.add_shape_box(
        body=-1, xform=wp.transform(wp.vec3(*table.pos), wp.quat_identity()),
        hx=float(table.half[0]), hy=float(table.half[1]), hz=float(table.half[2]),
        cfg=cfg, color=wp.vec3(*table.color), label="table")


def add_soft_block(builder: newton.ModelBuilder, cfg: SoftBlockConfig, center_pos) -> None:
    """Passive FEM block, centered in x/y at ``center_pos`` and resting on ``center_pos[2]``."""
    builder.default_particle_radius = cfg.particle_radius
    builder.particle_max_velocity = 50.0
    dx, dy, dz = cfg.dim
    builder.add_soft_grid(
        pos=wp.vec3(float(center_pos[0] - 0.5 * dx * cfg.cell),
                    float(center_pos[1] - 0.5 * dy * cfg.cell), float(center_pos[2])),
        rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=dx, dim_y=dy, dim_z=dz, cell_x=cfg.cell, cell_y=cfg.cell, cell_z=cfg.cell,
        density=cfg.density, k_mu=cfg.k_mu, k_lambda=cfg.k_lambda, k_damp=cfg.k_damp)


def add_cable(builder: newton.ModelBuilder, node_positions, cable: CableConfig = CABLE):
    """VBD rod (add_rod) with the shared cable material/damping. Returns (bodies, joints,
    body_start). Registers the authored contact material so the framework restores it after the
    blanket proxy-fill (the high cable friction the grasp relies on must survive the fill)."""
    cfg = newton.ModelBuilder.ShapeConfig(
        density=cable.density, margin=cable.contact_margin, ke=cable.contact_ke,
        kd=cable.contact_kd, mu=cable.friction)
    body_start = builder.body_count
    bodies, joints = builder.add_rod(
        positions=[wp.vec3(*p) for p in node_positions], radius=cable.radius, cfg=cfg,
        stretch_stiffness=cable.stretch_stiffness, stretch_damping=cable.stretch_damping,
        bend_stiffness=cable.bend_stiffness, bend_damping=cable.bend_damping,
        label="vbd_cable", wrap_in_articulation=True, body_frame_origin="start")
    _register_material(builder, bodies=bodies, ke=cable.contact_ke, kd=cable.contact_kd, mu=cable.friction)
    return bodies, joints, body_start


def add_rigid_box(builder: newton.ModelBuilder, pos, half, cfg: RigidBoxConfig, *,
                  mass: float | None = None, color=(0.18, 0.42, 0.95), label="cube", visible=True):
    """Single rigid box body. ``half`` is a scalar half-extent (cube). The body mass is ``mass`` if
    given, else ``cfg.mass`` (the canonical 1 kg cube), else density-derived. Returns (body, shape).
    Registers the authored contact material for the framework's post-blanket-fill restore."""
    if mass is None:
        mass = cfg.mass
    if mass is None:
        body = builder.add_body(xform=wp.transform(wp.vec3(*pos), wp.quat_identity()), label=label)
    else:
        body = builder.add_body(xform=wp.transform(wp.vec3(*pos), wp.quat_identity()), mass=mass, label=label)
    shape_cfg = newton.ModelBuilder.ShapeConfig(
        density=cfg.density, margin=cfg.contact_margin, ke=cfg.contact_ke, kd=cfg.contact_kd,
        mu=cfg.contact_mu, is_visible=visible)
    shape = builder.add_shape_box(
        body=body, hx=half, hy=half, hz=half, cfg=shape_cfg, color=wp.vec3(*color), label=f"{label}_shape")
    _register_material(builder, shapes=[shape], ke=cfg.contact_ke, kd=cfg.contact_kd, mu=cfg.contact_mu)
    return body, shape


def add_plate(builder: newton.ModelBuilder, pos, cfg: PlateConfig = PLATE):
    """The centralized presser plate: a thin sheet + a graspable handle on top, ONE rigid body.
    ``pos`` is the sheet center. Returns (body, [sheet_shape, handle_shape]). Registers the authored
    firm contact material for the framework's post-blanket-fill restore."""
    body = builder.add_body(xform=wp.transform(wp.vec3(*pos), wp.quat_identity()), label="plate")
    shape_cfg = newton.ModelBuilder.ShapeConfig(
        density=cfg.density, margin=cfg.contact_margin, ke=cfg.contact_ke, kd=cfg.contact_kd, mu=cfg.contact_mu)
    sheet = builder.add_shape_box(
        body=body, hx=float(cfg.sheet_half[0]), hy=float(cfg.sheet_half[1]), hz=float(cfg.sheet_half[2]),
        cfg=shape_cfg, color=wp.vec3(*cfg.sheet_color), label="metal_sheet")
    handle = builder.add_shape_box(
        body=body, xform=wp.transform(wp.vec3(*cfg.handle_local_pos), wp.quat_identity()),
        hx=float(cfg.handle_half[0]), hy=float(cfg.handle_half[1]), hz=float(cfg.handle_half[2]),
        cfg=shape_cfg, color=wp.vec3(*cfg.handle_color), label="grasp_handle")
    _register_material(builder, shapes=[sheet, handle], ke=cfg.contact_ke, kd=cfg.contact_kd, mu=cfg.contact_mu)
    return body, [sheet, handle]


def add_rubiks_cube(builder: newton.ModelBuilder, pos, cfg: RigidBoxConfig):
    """A rubik's cube: one collision box (invisible) plus a black body + 3x3 colored sticker grid
    per face (visual only). Returns the body index."""
    h = cfg.half_extent
    body, _ = add_rigid_box(builder, pos, h, cfg, label="rubiks_cube", visible=False)
    vis = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False)
    builder.add_shape_box(body=body, hx=h, hy=h, hz=h, cfg=vis, color=wp.vec3(0.03, 0.03, 0.03), label="cube_body")
    st, off = 0.024, h + 0.0008
    spacing = st * 2.0 / 3.0
    chs = st / 3.0 - 0.0013
    thin = 0.0006
    faces = [
        ((1, 0, 0), (0.80, 0.10, 0.10)), ((-1, 0, 0), (0.90, 0.45, 0.05)),
        ((0, 1, 0), (0.10, 0.20, 0.80)), ((0, -1, 0), (0.05, 0.55, 0.20)),
        ((0, 0, 1), (0.95, 0.95, 0.95)), ((0, 0, -1), (0.92, 0.85, 0.05)),
    ]
    for (nx, ny, nz), col in faces:
        for u in (-1, 0, 1):
            for v in (-1, 0, 1):
                if nx:
                    p, hxyz = wp.vec3(off * nx, u * spacing, v * spacing), (thin, chs, chs)
                elif ny:
                    p, hxyz = wp.vec3(u * spacing, off * ny, v * spacing), (chs, thin, chs)
                else:
                    p, hxyz = wp.vec3(u * spacing, v * spacing, off * nz), (chs, chs, thin)
                builder.add_shape_box(body=body, xform=wp.transform(p, wp.quat_identity()),
                                      hx=hxyz[0], hy=hxyz[1], hz=hxyz[2], cfg=vis,
                                      color=wp.vec3(*col), label="cube_cubie")
    return body


def add_ycb_mesh(builder: newton.ModelBuilder, cfg: YcbMeshConfig, pos, *, rest_on_z: float | None = None):
    """A YCB mesh object: coacd convex-hull pieces COLLIDE (consistent normals), the full mesh
    RENDERS (docs/SOLVERS.md §4). ``rest_on_z`` lifts the body so its lowest vertex sits on that z
    (e.g. the table top); otherwise ``pos[2]`` is used. Returns (body, mesh). Apply the realistic
    mass post-finalize via ``rescale_body_mass`` (the framework does this from the example's
    mass-override list)."""
    mesh = _mc.load_usd_mesh(OBJECTS_DIR / cfg.usd_subpath)
    z = float(pos[2])
    if rest_on_z is not None:
        z = float(rest_on_z) - float(np.asarray(mesh.vertices)[:, 2].min())
    body = builder.add_body(
        xform=wp.transform(wp.vec3(float(pos[0]), float(pos[1]), z), wp.quat_identity()),
        label=Path(cfg.usd_subpath).stem)
    _mc.add_collision_pieces(builder, body, mesh, cfg)
    _mc.add_visual_mesh(builder, body, mesh, color=cfg.color)
    return body, mesh
