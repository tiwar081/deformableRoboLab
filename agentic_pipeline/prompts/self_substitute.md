A scene is being RE-STAGED with object SUBSTITUTIONS enabled, but the human did not say which
swaps to make — you decide, and you write the substitution phrase they would have written.

The SCENARIO (unchanged):
$prompt

The objects currently in the scene:
$layout

The task the current scene affords (a good swap opens up a DIFFERENT kind of task):
$goal

The catalog you may swap in from:
$catalog

Produce "substitutions": $min_swaps to $max_swaps short phrases, each asking to swap ONE object in
the scene for ONE catalog object, e.g. "use a mug instead of the bowl" or "store the parts in a
woven tote bag instead of the tool bin". Rules:

- A substitution REPLACES an object one-for-one. It never adds or removes one — the object count
  stays exactly the same.
- The swap must keep the scenario plausible, and ideally enables a task the current object set
  cannot support (a deformable bag, garment, cable, or squishy item in place of a rigid object is
  an especially good swap for this simulator).
- A bag or garment can only be swapped INTO a scene of at most $cloth_max objects total (hard
  physics limit). This scene has $n_objects. If that rules a cloth swap out, pick a rigid,
  cable, or squishy swap instead.
- The RESULT of your swaps (the swapped-in objects together with everything already in the scene)
  must satisfy: $deformable_rule Never propose two swaps that each bring one of those in.
- Do not swap in an object the scene already contains, and only name objects by their catalog names
  (in the phrase, a natural article/noun form of the name is fine: "a mug", "the woven tote bag").

Respond with the JSON only.
