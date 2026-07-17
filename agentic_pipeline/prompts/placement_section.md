The robot is always mounted at the default position on its own ROBOT TABLE (a $stand_w m wide x
$stand_d m deep stand). You place the robot table so its front edge TOUCHES one edge of the
workspace table (never overlapping — the touch is built into the math). A placement is just:
- "edge": which workspace-table edge the robot stands at — one of $edges.
  Edge labels are fixed table labels: "back" = +y long edge, "front" = -y long edge,
  "left" = +x short edge, "right" = -x short edge.
- "anchor": the world coordinate ALONG that edge where the robot table is centred (x for the
  long back/front edges, y for the short left/right edges). The robot faces the table center-ward,
  perpendicular to its edge. Edge spans: back/front x in [$x0, $x1]; left/right y in [$y0, $y1].
  The robot table is $stand_w m wide, so an anchor near an end overhangs it (allowed) but must
  keep >= $min_contact m of actual edge contact.

Reach: the arm reaches ~$reach m from its base; the base sits ~$base_off m outside the chosen edge
at the anchor. An object is REACHABLE when some part of it (nearest point of its footprint) is
within that radius. Reference table — which scene objects are reachable from each candidate
placement (edge, anchor):
$reach_table

$placement_mode
