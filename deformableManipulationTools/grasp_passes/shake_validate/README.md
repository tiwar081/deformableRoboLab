# `shake_validate` — measuring a grasp candidate by grasping it

An offline **annotator** pass. It produces no candidates: for every upstream candidate it builds one
simulation in *this* simulator, puts a free-floating Franka hand on the candidate's pose, closes it
through the real force-feedback admittance controller, switches gravity on, shakes the hand, and
writes what happened into the five `grasp_library.QUALITY_FIELDS` with `quality_source` naming this
pass.

Since v4 (2026-08-11) that "every upstream candidate" is narrowed at TRIAL SELECTION, with the trial
itself unchanged: `retreated` (weak) and `seat_blocked` candidates are never trialed (no new
annotation — not chosen to run is different from unreachable), and annotation is INCREMENTAL — a
candidate whose id + pose + width match the pass's own previous sidecar carries its annotation
forward (`[pose:<digest>]` notes token) instead of re-simulating. Spec and rationale:
`docs/trajPipeline/grasp-passes.md` "Pass v4"; v3 sidecars stay mergeable via
`merge.COMPATIBLE_VERSIONS`.

```bash
# run the pass (writes _passes/shake_validate/<asset>.json, then merges)
.venv/bin/python -m deformableManipulationTools.grasp_passes run shake_validate
.venv/bin/python -m deformableManipulationTools.grasp_passes run shake_validate --asset sugar_box

# the pass's own tools
.venv/bin/python -m deformableManipulationTools.grasp_passes.shake_validate --selfcheck
.venv/bin/python -m deformableManipulationTools.grasp_passes.shake_validate --trial sugar_box
```

| file | what is in it |
|---|---|
| `__init__.py` | the `GraspPass` — trial results in, annotations out |
| `protocol.py`  | the test itself as data: the timeline, the shake, the force-target derivation, the thresholds |
| `hand.py`      | the free-floating gripper: the real Franka hand on a position-controlled 6-DOF base |
| `rig.py`       | one trial — a `framework.GraspExample` subclass, plus the metrics |
| `selfcheck.py` | the claims this pass makes, executable |

## The three commitments

**A free-floating gripper, not the arm.** A candidate is a property of the jaws and the object.
Running the arm would fold reachability — and the still-open end-effector rotation projection
question — into a number meant to say "does this pinch hold". ACRONYM splits it the same way: "the
gripper itself is simulated as an unconstrained position-controlled object".

`hand.py` does not model a gripper; it lifts the real one out of `robot.build_franka_robot` — the
same finger bodies, the same collision geometry, the same prismatic joints carrying
`FRANKA.finger_target_ke/kd`, `finger_effort` and `armature` — and re-parents it to the world through
a stiff 6-DOF joint. The subtree is found through `FRANKA`'s own link suffixes, so it follows
whichever robot `settings.yaml` selects. The base joint's frame **is** the grasp frame, which is why
placing a candidate needs no IK and why the shake is expressible in ACRONYM's own words: "up and
down along its approach direction" is base coordinate 2, "rotates around a line parallel to the
prismatic joint axes of the fingers" is base coordinate 3.

**The real closing dynamics.** The jaws are closed by the centralized `grip.GripController` driving
the real finger actuators against the real harvested squeeze — never by a scripted width. This is
why the rig takes the split MuJoCo+VBD route for *every* object, rigid ones included: the rigid-only
path has no gripper proxies, so no squeeze signal, so `force_stop_enabled == 0`, and the controller
degrades to a smoothstep to a fixed width. A candidate validated under that would tell you nothing
about the controller that actually runs.

Everything else about the physics is inherited rather than reimplemented: `rig.GraspTrial` subclasses
`framework.GraspExample` and calls its `_build_split_mujoco_vbd` and `_simulate_split`, so the
finalize ordering, the authored-material restore, the central harmonic particle-contact coupling, the
per-deformable VBD particle config, the realistic-mass override, the proxy build and the two-way
harvest are all the demo path's, unchanged. The pass supplies only a scene (one object) and a policy
(hold still, shake on schedule).

**Failures are recorded, never deleted.** A dropped candidate gets `object_in_gripper = 0`, a
`dropped` label and its motion numbers. A trial that fails *as a trial* gets `object_in_gripper = 0`,
a `diverged` label, null motions, and the reason in its notes. The pass owns no candidates and
removes none.

## The timeline

| phase | duration | gravity | what happens |
|---|---|---|---|
| settle  | 0.2 s | off | jaws at the Panda's full 80 mm aperture, on the candidate pose |
| close   | **until gripped**, ≤ 6 s | off | the admittance controller closes to the derived force target |
| ⟵ *`object_motion_during_closing_*` measured across this* | | | |
| load    | 0.5 s | **on** | gravity along the candidate's approach axis; the grasp takes the weight |
| shake ↕ | 2.0 s | on | 4 cycles at 2 Hz, ±40 mm along the approach |
| shake ↻ | 2.0 s | on | 4 cycles at 2 Hz, ±0.5 rad about the jaw axis |
| hold    | 0.5 s | on | quiescent; retention is judged here |
| ⟵ *`object_motion_during_shaking_*` measured from the end of the close to the end* | | | |

**The close ends on convergence, not on a clock**, and everything after it is anchored to whenever
that happens. This is ACRONYM's own wording — "fingers are closed using a velocity-based controller
until a force threshold is reached or the hand is fully closed" — and skipping it was a real bug
found while tuning. The admittance regulator's closing rate is proportional to the force *error*, so
a light object with a 1 N target closes its last millimetre roughly twenty times slower than a 20 N
one. A fixed 1.8 s window was ample for the 514 g sugar box and cut the 6 g sponge off mid-close, at
0.5 N of a 1 N target with the pads still 1.6 mm short of its surface — which then read as "dropped"
when the truth was "never gripped". With the close allowed to finish, the same candidate holds.
Convergence is read off the controller's own state (its filtered squeeze inside its own derived
deadband, held for 0.3 s); a candidate that never gets there is loaded anyway at the 6 s timeout and
its notes say `grip did NOT reach its N target`.

Two departures from ACRONYM, both deliberate:

*Gravity exists.* ACRONYM has none — it shakes precisely so that no gravity direction is assumed. We
were asked for both, so gravity is placed where it cannot corrupt the closing measurement: **off**
while the jaws close (an object in free fall would make `object_motion_during_closing_*` a
measurement of falling), **on** for the whole disturbance phase. Its direction is the candidate's own
approach axis — equivalent to rotating the world so the object hangs out of the jaws. That is the
standard pick load, and it is identical in gripper-relative terms for every candidate, so two
candidates' numbers compare instead of encoding which way the object happened to face.

*The shake is Hann-windowed.* A bare sine has peak velocity at t = 0, so the shake would open with a
velocity step the position-controlled base cannot follow, and the grasp would be tested by a start
transient whose size depends on base gains. The envelope leaves displacement and velocity zero at
both ends and puts peak acceleration in the middle of the window.

## What the five fields mean here

| field | measured as |
|---|---|
| `object_in_gripper` | 1.0 if both pads are pressing for ≥ 80 % of the final hold, else 0.0 |
| `object_motion_during_closing_linear` [m] | object centre displacement, start of close → end of close |
| `object_motion_during_closing_angular` [rad] | rigid rotation angle over the same interval |
| `object_motion_during_shaking_linear` [m] | object centre displacement, end of close → end of trial |
| `object_motion_during_shaking_angular` [rad] | rigid rotation angle over the same interval |

**Retention.** ACRONYM's test is "whether the object is still in contact with both fingers". The
literal reading does not survive contact with this engine: Newton's narrow phase emits contact
candidates out to the shape margin, so a pad reports "contacts" with an object it is 17 mm clear of
(measured, at the pre-grasp). The signal that means what the sentence means is the one the grip
controller already regulates — `TwoWayProxyCoupling.grip_squeeze_signal`, the **min** over the two
pads of the harvested reaction projected onto the closing axis. It is positive exactly when both pads
are pressing.

**"Rigid motion" of a deformable.** A deformable (an FEM body; historically also the cable rod)
has no pose, so the angular metric is
the *rigid component* of the motion: the Kabsch best-fit rotation of the object's point cloud
(its particles) against its configuration at t = 0, with the centroid for the
linear metric. A single rigid body reports its own COM and orientation exactly.

**A dropped candidate's shaking motion is a floor, not a measurement.** The trial stops as soon as
the object's centre passes `ESCAPE_RADIUS` (0.15 m) — everything after that is free fall, and letting
the clock run would replace "how far did it slip" with "how long was the trial". So a dropped
candidate reports ≈ 0.15 m; the number that matters for it is `object_in_gripper = 0`.

## The grasp force

`GraspWindow.force_target` is the one grasp knob a policy owns, and this pass is a policy, so it has
to choose one. Hand-picking per object would make candidates incomparable and would be exactly the
per-object tuning the codebase forbids, so it is derived — from the built model, not from a table:

```
force_target = clamp(SAFETY · m · (g + a_peak) / (2 · mu_eff),  1 N,  40 N)
```

the Coulomb squeeze two pads need to carry the object's peak inertial load. Every input is read off
the *finalized* model — mass after the realistic-mass override, friction after the central material
coupling (Newton mixes shape friction geometrically, so `sqrt(GRIP.proxy_mu · mu_object)`) — and
`a_peak` comes from the protocol's own peak accelerations with the object's largest half-extent as
the lever for the rotational shake.

`SAFETY` is where this simulator's behaviour enters, and it is why ACRONYM's constants could not be
imported even if they were published. See Tuning below.

## Tuning

Everything below was measured against this solver on the fixture candidates. Nothing is carried over
from ACRONYM, which publishes no amplitudes or thresholds and whose numbers came out of FleX.

**Shake amplitude.** 40 mm and 0.5 rad at 2 Hz peak at 6.3 m/s² and 79 rad/s². Chosen so the
disturbance is comparable to gravity rather than dominated by it (a shake much below g would just
re-measure the load phase) while staying an order of magnitude below the base actuator's corner
frequency, so the motion the object receives is the motion that was commanded. Shake fidelity is
reported per trial as achieved/commanded amplitude (`shake applied at X %/Y % of command` in the
pass notes; healthy trials read ~96 %/96 %). A raw base-tracking-error metric was retired: at 2 Hz
it is dominated by phase lag, which looks alarming (55 mrad on the can) and means nothing — while a
*partial* ratio is diagnostic (96 %/19 % = dropped part-way through the angular shake).

**Base gains** (`hand.py`) are stiff on purpose: the base is the fixture, not the subject. Sized so
the ~1 kg hand tracks the protocol's peak acceleration to well under the smallest motion reported.

**`FORCE_SAFETY` = 2, and it is NOT tuned to maximise the pass rate.** The obvious plan — sweep the
safety factor upward until geometrically sound grasps stop failing — does not work here, and finding
out why was the useful part of the tuning. Measured on the fixture candidates:

| candidate | safety 2 | safety 5 | safety 10 |
|---|---|---|---|
| `tomato_soup_can/mid_across_z` (smooth cylinder) | **held**, 11 mm slip | dropped @ 5.7 s | dropped @ 5.5 s |
| `sugar_box/mid_across_thin` (flat faces) | held, 3.6 mm | held, 3.1 mm | held, 3.1 mm |
| `banana/mid_span_side` (curved wedge) | — | held, 1.8 mm | held, 0.3 mm |

Squeezing **harder makes a round object worse**: at 31 N the can is ejected from a grasp that holds
at 12 N. That is the same mechanism `docs/physicsEngine/SOLVERS.md` §6 records for the banana wedge — a deeper
penalty penetration on a curved surface tilts the contact normal until the object squirts out, which
is also why "raising mu does not help" there. The banana meanwhile improves monotonically with force.
So the force target is not a reliability knob that can be turned up; there is no single value that
makes every sound grasp hold.

Given that, tuning it to a number would be tuning it to a *result*. Instead it stays at what physics
says — twice the Coulomb minimum for the load the protocol actually applies — which is a policy a
sensible controller would choose without knowing the answer. Candidates are then compared under
equal *relative* grip rather than under a constant chosen to flatter them. The consequence to be
aware of when reading the numbers: a `0` can mean the pose is bad **or** that this contact model
ejects that shape at a physically reasonable squeeze, and the pass does not distinguish the two.

**Repeatability.** Three repeated trials of each of three candidates produced **bit-identical**
metrics (agreement better than 10 µm and 1 µrad), and
`run shake_validate --asset sugar_box --check-idempotent` reports `ok sugar_box idempotent` — two
full runs, byte-identical sidecars. This matters more than it looks: the trial now
takes data-dependent branches — when the close converges, when the object escapes — and a
non-reproducible solve would move those decisions between runs and change the sidecar wholesale, not
just in its last digit. Reported values are nonetheless quantised (`ROUND_LINEAR`, `ROUND_ANGULAR`)
as insurance: the quanta are two orders below any resolution a consumer would act on, so they cost no
signal. They cannot rescue a value landing exactly on a quantum boundary — if `--check-idempotent`
ever fails, that is a rounding artefact rather than a changed measurement, and the fix is the quantum.

## What it said about the fixture candidates (2026-08-05 — HISTORICAL: v1 poses)

Full run over the six fixture assets — **5 held, 10 dropped, 1 diverged** of 15. These trials ran
on the pre-pad-seat v1 poses; the stored poses have since moved twice (v2 seating, then the
clamp/retreat) and the sidecars were deleted, so the NUMBERS are not current — the failure modes
and the discriminating power are the durable content. Worth reading not as
a verdict on the pass but as a demonstration that the numbers discriminate, since `fixture` says of
itself that its entries are "plausible and geometrically valid, but not validated grasps":

| candidate | held | closing motion | shaking motion | what the numbers say |
|---|---|---|---|---|
| `banana/mid_span_side`      | ✔ | 0.4 mm | 12.7 mm / 40° | well centred; gripped in 5.1 s at its 2.8 N target |
| `banana/mid_span_top`       | ✘ | 30.7 mm | — | mis-centred: material spans −5.7…+34.3 mm about the grasp centre, so one jaw punts it |
| `banana/near_end_top`       | ✘ | **150 mm** | — | punted clean out of the jaws *during the close* |
| `sugar_box/mid_across_thin` | ✔ | 1.3 mm | 3.1 mm / 10° | the best grasp in the set |
| `sugar_box/near_end_across_thin` | ✔ | 0.8 mm | 70 mm / 87° | holds, but slips 20× more than the mid-span one — the graded signal these fields exist for |
| `tomato_soup_can/mid_across_z` | ✔ | 0.8 mm | 10 mm / 47° | holds; the 47° is the can *rolling* between the pads, which a cylinder gripped across its diameter can do |
| `tomato_soup_can/mid_across_y` | ✘ | 0.6 mm | — | closed cleanly, then let go 2.8 s into the shake |
| `tomato_soup_can/near_end_across_y` | ✘ | 0.4 mm | — | gripped near the end of the barrel; gone 0.27 s after the load |
| `sponge/mid_across_thin`    | ✔ | 23 mm | 48 mm / 50° | a 6 g soft body: holds, but it is dragged 23 mm while the jaws close |
| `sponge/near_end_across_thin` | ✘ | 10 mm | — | same object, different pose, different answer |
| `power_cable/*` (3)         | ✘ | 11–16 mm | — | see the fingertip-plane limit below: only 7.3 mm of a 16 mm cable is inside the jaws (cables have since left the pipeline's scope, 2026-08-11) |
| `mug/rim_wall_px`           | — | null | null | **diverged** — the pads start inside the mug wall, and it is ejected at 9 m/s |
| `mug/rim_wall_py`           | ✘ | 150 mm | — | same pre-grasp overlap, resolved by punting instead |

Two of these are findings for whoever writes the real generators rather than for this pass: a grasp
centre must sit where the *material* is centred along the jaw axis (the banana's two failures), and
it must sit deep enough along the approach for the object to be inside the pads (the historical
cable fixtures' three — cables left the pipeline's scope 2026-08-11, but the seating lesson stands).

## Cost

Roughly 60–110 s per candidate on an A100. About a third is model build and MuJoCo compile: the base
joint's frame is baked at finalize, so each candidate needs its own model. The rest scales with how
long the close takes — a grasp that converges quickly is cheap, one that runs the 6 s timeout is not
— and trials that drop stop early. The six fixture assets' fifteen candidates take about 20 minutes.
Sidecars are cached the usual way, so a re-run costs nothing unless the pass version, the mesh, or
the upstream changed.

## Known limits

* **The grasp centre sits at the fingertip plane, and that decides a lot.** Measured on the panda
  hand: in the grasp frame the pads span the approach axis from −54.5 mm to −0.7 mm, so the TCP is
  0.7 mm past the very tip and the *entire* pad lies behind the grasp centre. That is
  `grasp_library`'s `POSE_CONVENTION` working as documented — the TCP is the point `WP.pos` commands,
  and `pickplace_ycb_vbd_franka` duly grips a 58 mm cube with the TCP at the cube's centre, taking
  its upper half. The rig honours it exactly; sliding the hand back to centre the object on the pads
  would validate a *different* pose than the one stored. The consequence for anyone reading the
  numbers: **a candidate is only as good as the material behind its origin.** The 16 mm `power_cable`
  centred on its own axis had 7.3 mm of itself inside the jaws and the rest hanging past the
  fingertips, and it was reported dropped largely for that reason (measured before cables left the
  pipeline's scope, 2026-08-11 — kept because the lesson is object-agnostic). That is a finding for
  whoever authors candidates — place the grasp centre deeper along the approach — not a defect here.
* **The pass says nothing about reachability.** By construction. A candidate that holds here may be
  unreachable for an arm in a real scene.
* **One gravity direction per candidate**, along its own approach. That is the standard pick load and
  it is fair across candidates, but it is not a search over orientations — ACRONYM's shake-only,
  gravity-free protocol is the alternative, and this is a deliberate departure from it.
* **The pre-grasp IS collision-checked** (added 2026-08-05, pass v2; the test —
  `grasp_library.pregrasp_collision` — moved into `grasp_library` 2026-08-06 so `pad_seat`'s
  collision-aware retreat and this check share one definition of the hand). Before any
  trial, the object's rest geometry is tested against the hand's own colliders — both fingers and
  the palm — at the pre-grasp opening the rig starts from, with the hand posed on the candidate.
  Material between the open jaws is the grasp; material inside a finger solid or the palm means the
  hand would have had to pass through the object to get there, so the candidate is **skipped, not
  simulated**. A skip carries labels (`shake_skipped`, `pregrasp_collision`) and **no quality and no
  `quality_source`** — nothing was measured, and an unmeasured `object_in_gripper = 0` would be
  indistinguishable from a measured one. This is what now separates "bad pinch" from "unreachable".

  Two things worth knowing about it. The colliders are tested as their true **convex hulls**, not
  bounding boxes: they are `CONVEX_MESH` shapes so the hull is exact, and the palm's AABB spans
  204x63x92 mm around a much smaller solid — with boxes the check rejected 52 % of banana and 99 %
  of mug candidates on palm slack alone. And the volumes are **measured off the active robot** at
  import, never hardcoded, because `settings.yaml` chooses the robot and the two supported hands
  differ in exactly this geometry.
* **Cloth and bags are out of scope**, upstream: they have no persistent rest shape, so there is no
  canonical frame to store a candidate in (`grasp_library.py`, SCOPE).
