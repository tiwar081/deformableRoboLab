"""Centralized object/asset builders. Every demo adds its scene objects through these so the
collision/representation NUANCES (mesh-vs-viz split, coacd convex decomposition for concave
meshes, FEM grids, contact materials, realistic masses) live in ONE place and a new demo cannot
re-introduce a per-object collision bug (e.g. the raw-concave-mesh ejection, docs/SOLVERS.md §4).

An example only chooses WHICH objects to add and WHERE (the scene); the physical properties come
from :mod:`deformableManipulationTools.params`.
"""
from __future__ import annotations

from dataclasses import dataclass
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
    # NOTE: builder.particle_max_velocity is INERT under SolverVBD (only XPBD/MPM/the base
    # integrator honor it). It is not set here so nothing relies on a non-existent velocity
    # ceiling — particle stability must come from the contact material (ke/kd), not a cap.
    dx, dy, dz = cfg.dim
    builder.add_soft_grid(
        pos=wp.vec3(float(center_pos[0] - 0.5 * dx * cfg.cell),
                    float(center_pos[1] - 0.5 * dy * cfg.cell), float(center_pos[2])),
        rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=dx, dim_y=dy, dim_z=dz, cell_x=cfg.cell, cell_y=cfg.cell, cell_z=cfg.cell,
        density=cfg.density, k_mu=cfg.k_mu, k_lambda=cfg.k_lambda, k_damp=cfg.k_damp)


@dataclass(frozen=True)
class ClothConfig:
    """T-shirt cloth (``add_cloth_mesh`` from the vendored ``assets/objects/unisex_shirt.usd``).

    EXPERIMENTAL — this is a thin SHELL, not a compressible FEM volume, and these values are adapted
    from Newton's ``example_cloth_franka`` (which runs in CENTIMETRE scale); they are NOT yet tuned for
    this framework's metre scale or the proxy grip. Exposes the same ``soft_contact_*`` fields the
    framework reads off ``soft_block`` so the proxy<->particle harvest works like the FEM-block demos."""
    usd_file: str = "unisex_shirt.usd"
    usd_prim: str = "/root/shirt"
    scale: float = 0.01               # the USD shirt mesh is ~65 cm in native units -> metres
    flatten_z: float = 0.12           # squash the native 3D (worn) shirt mesh in z so it starts LAID
                                       # FLAT on the table (~3 cm thick), instead of a 27 cm-thick draped
                                       # shape that would envelop the gripper proxies and drag the arm.
    density: float = 0.3              # per-area [kg/m^2] (~0.13 kg over the shirt)
    particle_radius: float = 0.005
    tri_ke: float = 1.0e4             # stretch / shear (cloth in-plane)
    tri_ka: float = 1.0e4
    tri_kd: float = 1.0e-2
    edge_ke: float = 5.0              # bending
    edge_kd: float = 0.5
    contact_margin: float = 0.01
    # proxy<->particle + body<->cloth contact (read by the framework like SoftBlockConfig).
    # CRITICAL for a thin shell: the ultra-light cloth particles (~3e-5 kg) have no volumetric tet
    # network or internal damping to absorb a contact impulse (unlike the FEM soft block, which masks
    # an under-damped contact), so the contact itself must be physically damped or the pads eject the
    # particles to NaN. Critical damping kd_crit = 2*sqrt(ke*m) ~ 1-4 N*s/m here; 1e1 is safely
    # over-critical and matches Newton's own example_cloth_franka (soft_contact_ke=1e4, kd=1e1). The
    # old ke=1e5/kd=1e-4 (carried over from the firm soft-block contact) was ~5 orders below critical
    # and over-stiff for this mass -> ejection. NOTE: the grip-force harvest (grip.py) reconstructs the
    # pad reaction from soft_contact_ke; cloth_franka wires coupling_soft_ke = CLOTH.soft_contact_ke, so
    # ke stays consistent across model penalty / VBD solve / harvest from this single source.
    soft_contact_ke: float = 1.0e4
    soft_contact_kd: float = 1.0e1
    soft_contact_kf: float = 1.0e3
    # Cloth (particle↔body) friction. Matched to Newton's example_cloth_franka, which sets
    # model.soft_contact_mu = 0.25 for the shirt (was 0.8 here). The effective cloth↔pad friction is
    # the VBD geometric mean √(soft_contact_mu · pad μ); the pad μ (GRIP.proxy_mu=1.0) is unchanged.
    soft_contact_mu: float = 0.25
    # ---- VBD particle self-contact (thin-shell fidelity; cf. cloth_solver_kwargs) ----
    # A thin shell folded/draped on itself MUST self-collide or layers pass through each other (and
    # through the table) — the FEM block never needs this (its volume can't fold), which is why the
    # shared PARTICLE_SOLVER_KWARGS leaves self-contact OFF. Newton's example_cloth_franka enables it;
    # these are its (cm-scale) values converted to this framework's metre scale. The topological filter
    # threshold + rest-shape exclusion radius stop mesh-adjacent vertices from self-colliding at rest.
    self_contact: bool = True
    self_contact_radius: float = 0.002       # [m] particle self-collision radius
    self_contact_margin: float = 0.003       # [m] broadphase margin for self-contact
    self_contact_filter_threshold: int = 1   # topological hops excluded from self-contact
    self_contact_rest_exclusion_radius: float = 0.006  # [m] rest-shape neighbours excluded


def cloth_solver_kwargs(cfg: ClothConfig) -> dict:
    """Centralized SolverVBD kwargs for a CLOTH scene (counterpart to PARTICLE_SOLVER_KWARGS, which is
    the FEM-block set). Keeps the thin-shell solver config — chiefly particle SELF-contact, which the
    block set omits — in the package, so a cloth demo declares only its scene, not solver physics."""
    return dict(
        rigid_body_contact_buffer_size=2048,
        rigid_body_particle_contact_buffer_size=16384,
        particle_enable_tile_solve=False,
        particle_collision_detection_interval=-1,
        particle_enable_self_contact=cfg.self_contact,
        particle_self_contact_radius=cfg.self_contact_radius,
        particle_self_contact_margin=cfg.self_contact_margin,
        particle_topological_contact_filter_threshold=cfg.self_contact_filter_threshold,
        particle_rest_shape_contact_exclusion_radius=cfg.self_contact_rest_exclusion_radius,
        particle_vertex_contact_buffer_size=32,
        particle_edge_contact_buffer_size=64,
    )


def add_cloth(builder: newton.ModelBuilder, cfg: ClothConfig, center_pos, *, yaw: float = 0.0):
    """Load the vendored T-shirt USD mesh and add it as a VBD cloth shell, centred in x/y at
    ``center_pos`` with its centroid at ``center_pos[2]`` (drop it slightly above the table and let it
    settle). EXPERIMENTAL (see :class:`ClothConfig`). Returns ``(particle_start, particle_count)``."""
    import newton.usd
    from pxr import Usd

    stage = Usd.Stage.Open(str(OBJECTS_DIR / cfg.usd_file))
    mesh = newton.usd.get_mesh(stage.GetPrimAtPath(cfg.usd_prim))
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    verts = verts - verts.mean(axis=0)            # centre the mesh at the origin (its native frame is offset)
    verts[:, 2] *= cfg.flatten_z                  # lay the draped shirt flat so it starts clear of the gripper
    builder.default_particle_radius = cfg.particle_radius
    # See add_soft_block: particle_max_velocity is inert under SolverVBD; not set (false safety net).
    p_start = builder.particle_count
    builder.add_cloth_mesh(
        pos=wp.vec3(*[float(x) for x in center_pos]),
        rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(yaw)),
        scale=float(cfg.scale), vel=wp.vec3(0.0, 0.0, 0.0),
        vertices=[wp.vec3(*v) for v in verts], indices=[int(i) for i in mesh.indices],
        density=cfg.density, tri_ke=cfg.tri_ke, tri_ka=cfg.tri_ka, tri_kd=cfg.tri_kd,
        edge_ke=cfg.edge_ke, edge_kd=cfg.edge_kd, particle_radius=cfg.particle_radius)
    return p_start, builder.particle_count - p_start


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
