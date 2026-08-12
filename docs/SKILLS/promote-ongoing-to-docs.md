# Skill: promoting `docs/ONGOING.md` content into `docs/` and resetting it

ONGOING.md is a scratchpad, not an archive. When knowledge in it becomes durable, it moves into
the permanent docs and ONGOING.md shrinks back to just the live edge. This skill is the procedure
for that transfer. (For writing INTO the file, see [update-ongoing.md](update-ongoing.md).)

## When to run this

- A big task **completes** (or is stopped for good).
- A **new big task starts** — the file is reset per task, so promote the old task's residue first.
- The file has **bloated mid-task**: proven-and-durable sections are accumulating that no longer
  change. Promote those sections early; the in-flight remainder stays.

## Procedure

1. **Read the whole file** and classify every section:
   - *Durable* — proven results, measured numbers, conventions, scope boundaries, and dead ends
     that future work must not re-walk.
   - *Transient* — status snapshots, disproven hypotheses, superseded measurements whose successor
     carries the conclusion, coordination notes between agents. These are deleted, not moved.

2. **Pick a destination for each durable item.** Everything except ONGOING.md lives in a subfolder
   of `docs/`:
   - Match the topic to a folder — `physicsEngine/`, `agenticPipeline/`, `trajPipeline/`,
     `rendering/` — and prefer extending the existing doc that owns the topic over
     creating a new one (least overlap wins).
   - No doc fits → create one in the right subfolder and add it to the `docs/` index in AGENTS.md
     with a one-line description.
   - A rule every future session needs → AGENTS.md itself, sparingly: it has a strict instruction
     budget (see [writing-claude-md.md](writing-claude-md.md)). A mistake agents will otherwise
     repeat → its "Recurring mistakes to avoid" section.
   - `docs/project-overview.md` changes only on MAJOR pipeline restructuring — promotion
     of task results almost never touches it.

3. **Rewrite for the destination — don't paste.** Docs describe current state; ONGOING narrates a
   task. Drop the chronology and the "landed/measured on <date>" framing; keep the measured
   numbers, the "do not re-derive" facts, and the *why* behind each decision. A dead end is worth
   keeping only if recording it prevents re-walking it (the SOLVERS.md §6 trade study is the
   model); write it as "measured dead end — do not re-walk", with the numbers.

4. **Repoint references.** Grep the repo for `ONGOING.md` — code and docs cite specific sections
   (e.g. a module docstring citing the SCOPE section). Any reference whose target you are about to
   delete must be repointed to the promoted location before the reset.

5. **Reset ONGOING.md** to: the short header (what the file is + pointers to the two skills), then
   either the new task's `## In flight:` section or nothing. Raw data referenced from deleted
   sections lives in session scratchpads — if it still matters, the promoted doc keeps the
   pointer; otherwise it goes.

6. **Self-check**: every measured number and named convention from the deleted sections is either
   in a permanent doc or deliberately discarded as transient; no repo reference points at a
   deleted section; the AGENTS.md index lists any new doc.
