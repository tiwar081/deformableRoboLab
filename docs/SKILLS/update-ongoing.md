# Skill: updating `docs/ONGOING.md`

`docs/ONGOING.md` is the scratchpad for the **current** in-flight task: what's unresolved right
now, what was just tried, and the working hypotheses. It is the ONE doc that lives at the `docs/`
root, and it is the authority on active work — read it before editing an area it names as active,
and trust it over any summary elsewhere. Everything durable eventually leaves this file (see
[promote-ongoing-to-docs.md](promote-ongoing-to-docs.md)); this skill is about writing INTO it.

## When to write an entry

Update ONGOING.md immediately after any of these, not at the end of a session:

- **Something landed** — code merged into the shared packages, an artifact regenerated.
- **Something was measured** — an experiment, a sweep, a gate. Record numbers, not impressions.
- **Something was fixed** — with the defect kept alongside if its numbers explain the fix.
- **A decision was made without work** — e.g. "cables stay disabled in the generators, because…".
- **A gotcha was hit** that the next agent in this task would otherwise re-discover.
- **A new big task starts** — reset the file first (see the promote skill), then open with the
  task header and its SCOPE.

## Structure conventions (match the existing file)

- One top-level section per big task: `## In flight: <task> (starting YYYY-MM-DD)`.
- `### SCOPE` right under it when the boundary is non-obvious — state what the work does NOT
  cover and *why the boundary falls there*, so nobody extends it by analogy.
- Entry headings carry a status verb and an absolute date:
  `### Landed: <thing> (<file>)`, `### MEASURED (YYYY-MM-DD): <what>`,
  `### FIXED (YYYY-MM-DD): <what>`, `### Decisions recorded (no work)`, and a final `### Next`
  queue of open items (with enough context that a fresh agent can pick one up).
- Keep a **"Measured while building — do not re-derive"** list per landed piece: the expensive
  facts (timings, tolerances, failure modes) that must not be re-measured or second-guessed.

## Rules that keep the file trustworthy

- **Absolute dates only** ("2026-08-06", never "today"/"yesterday") — entries outlive sessions.
- **Say how you know**: "verified by reading", "measured, N=51", "swept from disk". Agent reports
  go stale — a pass reporting on shared state is reporting on whatever it last read. When counts
  matter, re-derive them from disk and say you did.
- **Supersede, don't delete, while the task is live.** Mark the old section
  `(SUPERSEDED by the above — <reason>)` in its heading. Old numbers stay comparable; deletion
  happens at promotion time.
- **Keep raw data out**: full trial rows, backups, and dumps go to the session scratchpad; the
  entry names the file (e.g. "full rows in `strat2_shake.json`") and keeps only the summary table.
- **Note concurrent agents.** If parallel work can move shared state under a measurement, record
  that (what moved, when) so the numbers are interpretable later.
- **Keep it lean.** Proven + durable → promote out and delete here; disproven hypotheses → delete
  outright. The file describes the task's live edge, not its history.
