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
                     SoftMeshConfig, TableConfig, YcbMeshConfig)
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
    shape = builder.add_shape_box(
        body=-1, xform=wp.transform(wp.vec3(*table.pos), wp.quat_identity()),
        hx=float(table.half[0]), hy=float(table.half[1]), hz=float(table.half[2]),
        cfg=cfg, color=wp.vec3(*table.color), label="table")
    # Register the authored material like every other builder (2026-07-11 fix): without this the
    # blanket proxy-fill silently replaced the table's material post-finalize — invisible while
    # the fill's mu (0.8) happened to equal the table's old authored 0.8, live the moment the
    # table was re-authored realistic (0.5).
    _register_material(builder, shapes=[shape], ke=table.object_ke, kd=table.object_kd, mu=table.object_mu)
    return shape


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

    The values are Newton's ``example_cloth_franka`` cloth converted UNIT-CONSISTENTLY to this
    framework's SI (m, kg) world. Newton's example runs in centimetre-gram units, so copying its
    numbers verbatim (the original state of this config) made the cloth contact ~1000x stiffer and
    ~100x more damped RELATIVE TO PARTICLE WEIGHT than Newton's working grasp, and the shirt 10x too
    light — which is what actually broke the cloth grasp (see docs/cloths.md). The conversion law
    (cm,g -> m,kg): stiffness-like [M/T^2] (ke, kf, tri_ke/ka) x1e-3; damping-like [M/T] (kd) x1e-3;
    bending [M*L/T^2] (edge_ke; force units, VBD's k = edge_ke*rest_len with dtheta/dx ~ 1/L) x1e-5;
    edge_kd [M*L/T] x1e-5; area density [M/L^2] x10 (0.02 g/cm^2 = 0.2 kg/m^2); lengths x0.01.
    Matching dt (10 substeps) keeps every contact dimensionless group EQUAL to Newton's: the
    framework derives the shape side of the cloth contact centrally (harmonic coupling, see
    framework.py), and the cloth's own ke/kd are authored (15 / 0.015) so that derivation lands
    exactly on the PROVEN effective values — ke_eff = harm(15, 5e4) = 30 N/m (eta = ke_eff*dt^2/m
    = 3.2, Newton's avg(10, 50)) and kd_eff = 0.03 = 0.5x critical. The effective contact is the
    tuned quantity: at authored 10/0.01 (eff 20 / eta 2.1) the marginal flat-sheet pinch sheds at
    lift — MEASURED 2026-07-10. Exposes the same ``soft_contact_*`` fields the framework reads
    off ``soft_block`` so the proxy<->particle harvest works like the FEM-block demos."""
    usd_file: str = "unisex_shirt.usd"
    usd_prim: str = "/root/shirt"
    # Optional IMPORTED garment mesh (path under assets/objects; .obj/.ply/.stl via trimesh, or a
    # .usd/.usda whose first mesh prim is taken). None -> the vendored shirt (usd_file/usd_prim).
    # Physical fields above/below stay the ONE cloth schema — an imported garment overrides only
    # what its source dataset provides (e.g. SimWeaver per-garment density/stiffness), via
    # dataclasses.replace on this config in the scene catalog.
    mesh_file: str | None = None
    # Some render-oriented garment meshes duplicate the vertices along sewn seams to keep UVs and
    # normals discontinuous.  Physics needs those coincident seam vertices to be shared so the
    # shell is one connected body.  Opt in only for assets verified to use that convention.
    weld_shared_vertices: bool = False
    # Optional post-weld topology invariant.  A T-shirt is one connected orientable shell with
    # four boundary loops: neck, bottom hem, and two sleeve cuffs.  Asset-specific validation keeps
    # a UV seam weld from silently sealing a real garment opening or leaving a panel detached.
    expected_boundary_loops: int | None = None
    scale: float = 0.01               # the USD shirt mesh is ~65 cm in native units -> metres
    flatten_z: float = 0.12           # squash the native 3D (worn) shirt mesh in z so it starts LAID
                                       # FLAT on the table (~3 cm thick); the cloth_franka demos pass 1.0
                                       # to drop the inflated shirt like Newton's init.
    density: float = 0.2              # per-area [kg/m^2] = Newton's 0.02 g/cm^2 (~0.17 kg shirt)
    particle_radius: float = 0.008    # Newton's 0.8 cm — the cloth's contact thickness vs bodies
    tri_ke: float = 10.0              # stretch / shear (Newton 1e4, [M/T^2] x1e-3)
    tri_ka: float = 10.0
    tri_kd: float = 1.5e-5            # Newton 1.5e-2 x1e-3
    edge_ke: float = 5.0e-5           # bending (Newton 5, force units x1e-5)
    edge_kd: float = 5.0e-6           # Newton 0.5 x1e-5
    contact_margin: float = 0.008     # cloth<->body pipeline margin = Newton's 0.8 cm
    # proxy<->particle + body<->cloth contact (read by the framework like SoftBlockConfig). These
    # are the CLOTH'S OWN contact properties; the shape side of the contact comes from each
    # touching shape's own authored material via the framework's central harmonic ke/kd coupling
    # (framework._build_split_mujoco_vbd) — no cloth-authored shape profile. ke_eff = 30 N/m
    # sounds soft in metre units but is exactly Newton's working value: penetration under the
    # shirt's own weight is ~10 um/particle, and an 8 mm jaw on the ~2-layer stack gives
    # sub-newton per-contact forces, not the old 1e2-N spikes that expelled the shell
    # ("watermelon-seed" ejection).
    soft_contact_ke: float = 15.0     # authored so the central HARMONIC coupling with rigid-scale
                                       # shapes (pads/table ke=5e4) lands on the PROVEN effective
                                       # 30 N/m: harm(15, 5e4) = 30 = Newton's avg(10, 50). The
                                       # effective contact (eta = 3.2) is the tuned quantity, not
                                       # the raw particle number — at authored 10 (eff 20, eta 2.1)
                                       # the marginal pinch sheds at lift, MEASURED 2026-07-10.
                                       # NOTE self-contact shares this scalar (x1.5 vs Newton's 10;
                                       # fold A/B-verified).
    soft_contact_kd: float = 1.5e-2   # same re-authorship for damping: harm(0.015, 1e2) = 0.03 =
                                       # Newton's effective, 0.5x critical. (The original verbatim
                                       # kd=1e1 was ~100x OVER-critical: viscous glue that ratcheted
                                       # particles out of the pinch during motion.)
    soft_contact_kf: float = 1.0      # Newton 1e3 x1e-3
    soft_contact_mu: float = 0.25     # dimensionless; the cloth side of every cloth<->shape pair
                                       # (geometric mix) AND the self-contact friction directly.
                                       # Effective vs the REALISTIC authored shapes: pads 0.8 ->
                                       # sqrt(0.25*0.8) ~ 0.45, table 0.5 -> ~ 0.35. (No shape mu
                                       # stamp anymore: under Newton's cheat friction the recipe
                                       # needed table eff ~0.6, and the fold recipe now compensates
                                       # the physical value with pressing normal force instead —
                                       # docs/cloths.md "grasp recipe".)
    # ---- VBD particle self-contact (thin-shell fidelity; cf. cloth_particle_kwargs) ----
    # A thin shell folded/draped on itself MUST self-collide or layers pass through each other (and
    # through the table) — the FEM block never needs this (its volume can't fold), which is why the
    # shared PARTICLE_SOLVER_KWARGS leaves self-contact OFF. Newton's example_cloth_franka enables it;
    # these are its values (cm -> m). The topological filter threshold + rest-shape exclusion radius
    # stop mesh-adjacent vertices from self-colliding at rest. The 2 mm self-contact radius is what
    # sets the settled two-layer spacing (measured ~2 mm in Newton's recording — there is NO large
    # inter-layer gap; the graspable thickness comes from particle_radius vs the bodies).
    self_contact: bool = True
    self_contact_radius: float = 0.002       # [m] particle self-collision radius (Newton 0.2 cm)
    self_contact_margin: float = 0.002       # [m] broadphase margin for self-contact (Newton 0.2 cm)
    self_contact_filter_threshold: int = 1   # topological hops excluded from self-contact
    self_contact_rest_exclusion_radius: float = 0.005  # [m] rest-shape neighbours excluded (Newton 0.5 cm)


# RGBench's measured green knit T-shirt.  This is a distinct garment from the
# canonical Newton shirt: its mesh, native scale, area density, and cloth-side
# friction belong to the asset, while the proven SI-converted shell/contact
# model above remains shared.
GREEN_TSHIRT = ClothConfig(
    mesh_file="rgbench/green_tshirt.obj",
    weld_shared_vertices=True,
    expected_boundary_loops=4,
    scale=0.65,
    flatten_z=1.0,  # preserve RGBench's authored geometry; only duplicate seam vertices are welded
    density=0.22,
    soft_contact_mu=0.27,
)

def cloth_particle_kwargs(cfg: ClothConfig) -> dict:
    """The per-deformable PARTICLE SolverVBD config for a CLOTH (thin shell) — chiefly particle
    SELF-contact, which the FEM-block set (PARTICLE_SOLVER_KWARGS) omits. This is applied CENTRALLY by
    the framework whenever a cloth is present (it auto-detects shell vs FEM from the finalized model),
    so a cloth demo never declares solver physics — adding a cloth to any scene needs no demo override.
    Counterpart to PARTICLE_SOLVER_KWARGS; both exclude the scene-specific rigid_body_* buffer sizes."""
    return dict(
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


def _validate_cloth_topology(tm, cfg: ClothConfig) -> None:
    """Validate an imported garment after seam welding.

    Boundary vertices must form disjoint degree-two loops.  This detects both kinds of bad repair:
    welding across a real opening reduces the loop count, while omitting a sewn seam adds loops or
    leaves multiple connected components.
    """
    if cfg.expected_boundary_loops is None:
        return

    edges = np.sort(np.asarray(tm.edges, dtype=np.int64), axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    nonmanifold = int(np.count_nonzero(counts > 2))
    boundary = unique_edges[counts == 1]

    adjacency: dict[int, list[int]] = {}
    for a, b in boundary:
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))
    bad_degrees = [v for v, neighbours in adjacency.items() if len(neighbours) != 2]

    loops = 0
    unseen = set(adjacency)
    while unseen:
        loops += 1
        stack = [unseen.pop()]
        while stack:
            for neighbour in adjacency[stack.pop()]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    stack.append(neighbour)

    components = int(tm.body_count)
    expected = int(cfg.expected_boundary_loops)
    if components != 1 or nonmanifold or bad_degrees or loops != expected:
        raise ValueError(
            f"invalid cloth topology for {cfg.mesh_file}: components={components}, "
            f"boundary_loops={loops} (expected {expected}), nonmanifold_edges={nonmanifold}, "
            f"irregular_boundary_vertices={len(bad_degrees)}"
        )


def add_cloth(builder: newton.ModelBuilder, cfg: ClothConfig, center_pos, *, yaw: float = 0.0):
    """Load the vendored T-shirt USD mesh and add it as a VBD cloth shell, centred in x/y at
    ``center_pos`` with its centroid at ``center_pos[2]`` (drop it slightly above the table and let it
    settle). EXPERIMENTAL (see :class:`ClothConfig`). Returns ``(particle_start, particle_count)``."""
    import newton.usd
    from pxr import Usd

    if cfg.mesh_file is not None and Path(cfg.mesh_file).suffix.lower() in (".obj", ".ply", ".stl"):
        import trimesh
        tm = trimesh.load(str(OBJECTS_DIR / cfg.mesh_file), force="mesh")
        if cfg.weld_shared_vertices:
            # RGBench's green shirt has five UV/normal-separated islands (front, back, two sleeves,
            # collar) whose intended seam vertices coincide exactly.  Merging them yields one
            # manifold shell while preserving its four real openings; the topology guard below
            # rejects any future asset/edit that instead seals an opening.
            tm.merge_vertices(merge_tex=True, merge_norm=True)
        _validate_cloth_topology(tm, cfg)
        verts = np.asarray(tm.vertices, dtype=np.float64)
        indices = np.asarray(tm.faces, dtype=np.int64).reshape(-1)
    else:
        stage = Usd.Stage.Open(str(OBJECTS_DIR / (cfg.mesh_file or cfg.usd_file)))
        prim = stage.GetPrimAtPath(cfg.usd_prim) if cfg.mesh_file is None else next(
            p for p in stage.Traverse() if p.GetTypeName() == "Mesh")
        mesh = newton.usd.get_mesh(prim)
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        indices = mesh.indices
    verts = verts - verts.mean(axis=0)            # centre the mesh at the origin (its native frame is offset)
    verts[:, 2] *= cfg.flatten_z                  # lay the draped shirt flat so it starts clear of the gripper
    builder.default_particle_radius = cfg.particle_radius
    # See add_soft_block: particle_max_velocity is inert under SolverVBD; not set (false safety net).
    p_start = builder.particle_count
    builder.add_cloth_mesh(
        pos=wp.vec3(*[float(x) for x in center_pos]),
        rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(yaw)),
        scale=float(cfg.scale), vel=wp.vec3(0.0, 0.0, 0.0),
        vertices=[wp.vec3(*v) for v in verts], indices=[int(i) for i in indices],
        density=cfg.density, tri_ke=cfg.tri_ke, tri_ka=cfg.tri_ka, tri_kd=cfg.tri_kd,
        edge_ke=cfg.edge_ke, edge_kd=cfg.edge_kd, particle_radius=cfg.particle_radius)
    return p_start, builder.particle_count - p_start


def load_tet_mesh(path):
    """Parse an IsaacGym-format ``.tet`` file (NVlabs DefGraspSim: ``v x y z`` vertices,
    ``t i0 i1 i2 i3`` 0-indexed tetrahedra). Returns ``(verts (N,3) float64, tets (M,4) int32)``."""
    verts, tets = [], []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                verts.append([float(x) for x in parts[1:4]])
            elif parts[0] == "t":
                tets.append([int(i) for i in parts[1:5]])
    if not verts or not tets:
        raise ValueError(f"{path} contains no tet mesh (v/t lines)")
    return np.asarray(verts, dtype=np.float64), np.asarray(tets, dtype=np.int32)


def add_soft_mesh_object(builder: newton.ModelBuilder, cfg: SoftMeshConfig, center_pos, *, yaw: float = 0.0):
    """Imported squishy object: a volumetric FEM body from a ``.tet`` file (DefGraspSim format),
    centred in x/y at ``center_pos`` and RESTING ON ``center_pos[2]`` (lowest vertex at that z),
    like :func:`add_soft_block`. The framework detects the tets (``model.tet_count``) and applies
    the FEM particle-solver config + this config's ``soft_contact_*`` centrally — a demo adds the
    object and nothing else. Returns ``(particle_start, particle_count)``."""
    verts, tets = load_tet_mesh(OBJECTS_DIR / cfg.tet_subpath)
    verts = verts * float(cfg.scale)
    center = verts.mean(axis=0)
    verts[:, 0] -= center[0]
    verts[:, 1] -= center[1]
    verts[:, 2] -= verts[:, 2].min()              # rest the lowest vertex on center_pos[2]
    builder.default_particle_radius = cfg.particle_radius
    p_start = builder.particle_count
    builder.add_soft_mesh(
        pos=wp.vec3(*[float(x) for x in center_pos]),
        rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(yaw)),
        scale=1.0, vel=wp.vec3(0.0, 0.0, 0.0),
        vertices=[wp.vec3(*v) for v in verts], indices=[int(i) for i in tets.reshape(-1)],
        density=cfg.density, k_mu=cfg.k_mu, k_lambda=cfg.k_lambda, k_damp=cfg.k_damp,
        particle_radius=cfg.particle_radius)
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


def add_ycb_mesh(builder: newton.ModelBuilder, cfg: YcbMeshConfig, pos, *, rest_on_z: float | None = None,
                 yaw: float = 0.0):
    """A YCB mesh object: coacd convex-hull pieces COLLIDE (consistent normals), the full mesh
    RENDERS (docs/SOLVERS.md §4). ``rest_on_z`` lifts the body so its lowest vertex sits on that z
    (e.g. the table top); otherwise ``pos[2]`` is used. ``yaw`` spins the body about world z.
    Returns (body, mesh). Apply the realistic mass post-finalize via ``rescale_body_mass`` (the
    framework does this from the example's mass-override list)."""
    mesh = _mc.load_usd_mesh(OBJECTS_DIR / cfg.usd_subpath)
    z = float(pos[2])
    if rest_on_z is not None:
        z = float(rest_on_z) - float(np.asarray(mesh.vertices)[:, 2].min())
    body = builder.add_body(
        xform=wp.transform(wp.vec3(float(pos[0]), float(pos[1]), z),
                           wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(yaw))),
        label=Path(cfg.usd_subpath).stem)
    hull_shapes = _mc.add_collision_pieces(builder, body, mesh, cfg)
    # Register the authored material (2026-07-11 fix): the blanket proxy-fill on the VBD path used
    # to silently replace it (the MuJoCo rigid-only path kept the build-time values — the two ycb
    # twins ran DIFFERENT object friction without anyone authoring that).
    _register_material(builder, shapes=hull_shapes, ke=cfg.ke, kd=cfg.kd, mu=cfg.mu)
    _mc.add_visual_mesh(builder, body, mesh, color=cfg.color)
    return body, mesh
