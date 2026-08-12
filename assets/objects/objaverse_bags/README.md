# Objaverse deformable bags

Three bags are vendored here—deliberately the same order of magnitude as the three garment assets
already in `scene_catalog.json`. They are a starter set, not an exhaustive import. More can be added
later by querying Objaverse annotations for `plastic_bag`, `grocery_bag`, `tote_bag`, or
`shopping_bag`, adding a data entry to `manifest.json`, and rerunning the generic converter.
The current source fetch used the official `objaverse==0.1.7` Python package (`load_lvis_annotations`,
`load_annotations`, then `load_objects` only after the per-object license check).

## Sources and licenses

Objaverse metadata uses the license code `by` for these objects. Each source is licensed
[Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/); attribution and
the immutable Objaverse UID are recorded below and inside each USDA.

| local asset | Objaverse UID | original title / author | annotation | license |
|---|---|---|---|---|
| `plastic_recycle_bag.usda` | `181124c146e94ea7a03022d0c9b0c8d8` | [Guusje van Hees](https://sketchfab.com/3d-models/181124c146e94ea7a03022d0c9b0c8d8) / OnlineExpo | `grocery_bag` | CC BY 4.0 (`by`) |
| `woven_tote_bag.usda` | `c7da540534134560af463fdd29ed209f` | [Tote Bag C Over Magnet](https://sketchfab.com/3d-models/c7da540534134560af463fdd29ed209f) / eeelabvisual | `shopping_bag` | CC BY 4.0 (`by`) |
| `paper_grocery_bag.usda` | `d6814ce433f64d0da49f4a8b1ca3bb00` | [Burger King Paper Bag](https://sketchfab.com/3d-models/d6814ce433f64d0da49f4a8b1ca3bb00) / RPSebb | `grocery_bag` | CC BY 4.0 (`by`) |

The original GLBs are retained under `source/` so conversion is reproducible. Textures and logos
are not transferred to the remeshed simulation shells; the USDA files use neutral preview colors.

## Conversion and physics

Run:

```bash
.venv/bin/pip install objaverse==0.1.7  # only needed to fetch another UID
.venv/bin/pip install pymeshlab==2025.7.post1
.venv/bin/python assets/objects/_utils/convert_bag_mesh.py \
  assets/objects/objaverse_bags/manifest.json
```

The converter applies scene transforms, welds exact duplicate vertices, keeps the largest connected
bag shell, removes surplus faces on non-manifold edges, converts Y-up to Z-up metres, and performs
an area-preserving lay-flat rotation followed by isotropic remeshing at a 12 mm target edge length.
Only the front/back separation is compressed; the large wall panels and their areal mass are not
shrunk. This gives an honest collapsed tabletop start—an unsupported cloth bag is not silently held
open by pressure or rigidity. The converter rejects disconnected output, non-manifold edges,
irregular boundaries, or an unexpected number of boundary loops. Detached render-only straps,
labels, and hardware are intentionally discarded: without a sewn constraint they would be separate
cloth bodies and fall away.

Objaverse supplies geometry and license metadata, not measured cloth parameters. The physical
values in `manifest.json` are therefore explicit **inferences** anchored to this repository's proven
SI `ClothConfig`: approximately 25 g/m² film, 200 g/m² woven fabric, and 80 g/m² paper. Stretch and
bending are conservative material-relative variants of the Newton cloth baseline. Particle-contact
`ke/kd/kf` are scaled with area density where mass changes substantially so the contact stiffness
relative to particle mass stays near the established cloth operating point. These are suitable
initial simulation parameters, not claims about measurements of the original Objaverse objects.

Each USDA carries standard USD `PhysicsMassAPI` and `PhysicsMaterialAPI` data plus the exact runtime
parameters as `newton:cloth:*` attributes. The same manifest values are registered as `ClothConfig`
arguments in `scene_catalog.json`; `add_cloth()` remains the runtime VBD authority.
