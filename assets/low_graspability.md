# Low-graspability ledger

Objects whose grasp the FULL PIPELINE has evidenced will fail consistently (written automatically
by `deformableManipulationTools/traj_gen/stage.py` when a trajectory aborts on object-bound,
placement-independent grasp evidence — library verdicts, jaw sweeps below the tabletop, or every
rollout failing at the grasp itself). Scene/task generation deliberately does NOT read this file
(or any grasp data); it exists so humans and the grasp-generator roadmap know where the gaps are.
One section per object; evidence accumulates per run.

## tuna_can
- 2026-08-12 `demo_kitchen_putaway` (task.json, task 'put_tuna_can_in_bowl'): candidate pool exhausted — 16 candidate(s) sweep the jaw below the tabletop (object geometry, placement-independent); the 2 rollout(s) run all failed the grasp itself (never_held)
- 2026-08-12 `demo_kitchen_putaway_b` (task.json, task 'put-tuna-can-in-bowl'): candidate pool exhausted — 17 candidate(s) sweep the jaw below the tabletop (object geometry, placement-independent); the 2 rollout(s) run all failed the grasp itself (never_held)

## green_tshirt
- 2026-08-13 `demo_laundry_full` (task_2.json, task 'retrieve_green_tshirt'): all 3 rollout attempts across different candidates failed at the grasp itself (never_held)
- 2026-08-13 SUPERSEDED: those failures were edge pinches at/above the cloth surface; with the
  measured sheet recipe in the online-grasp prompt (grasp 3-6 cm INSIDE the fabric, z 4-6 mm
  below the tabletop, 4-5 N, 2 s close + 3 s press-hold) the same task GRASPED and completed on
  its first proposal (`demo_laundry_full` retry, visually verified). Not low-graspability —
  recipe-sensitive.

## banana
- 2026-08-13 `demo_fruit_full` (task_3.json, task 'banana-beside-bowl'): candidate pool exhausted — 30 candidate(s) sweep the jaw below the tabletop (object geometry, placement-independent)
