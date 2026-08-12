You propose ONE meaningful, physically feasible manipulation task for a Franka-arm robot, AND you
decide where the robot stands. The scene (below) may contain RIGID and DEFORMABLE objects (cloth,
cables/ropes, bags, squishy FEM objects). You choose the goal, describe it, and place the robot; you do
NOT decide grasp force or trajectory.

SCENE OBJECTS (reference objects ONLY by these exact names):
$objects

GOAL PREDICATES (choose exactly one; its "object" param is the thing being manipulated):
$predicates

HARD RULES (the feasibility checker enforces them — violating one gets the task rejected):
- AFFORDANCE: the predicate must match the object's category. Cloth can fold/spread/drape/uncover;
  cable can coil/route/thread/straighten/avoid snags; squishy clutter and general objects can be
  pushed/pulled/retrieved/cleared; bag goals require a BAG. Do not fold a banana or stack a rope.
- CONTAINER: object_in_container / object_outside_of / sort need an OPEN-TOP container object
  (bowl, mug, pitcher) — a can, box, or block is not a container.
- FIT: to place an object into a container it must fit — a large cloth may need to be FOLDED first
  (fold it in your subtasks); if it cannot fit even folded, pick another goal.
- REACH: every object the task manipulates or references must have SOME PART within the robot's
  reach from the placement you choose (see ROBOT PLACEMENT below).

ROBOT PLACEMENT:
$placement_section

Also produce THREE instruction phrasings (default = clear and complete; vague = terse/underspecified;
specific = names colors/attributes), a SUBTASK list (ordered atomic steps, e.g. grasp -> fold ->
place), competency ATTRIBUTES, and a difficulty (simple|moderate|complex). Prefer tasks that exercise
the deformable objects when the scene has them. If a check rejects your proposal you will get the
concrete reach/alignment numbers back — you may then EITHER move the robot (new placement) OR pick a
different task; choose whichever preserves the better task. Respond with the task JSON only.
