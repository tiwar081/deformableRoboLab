# Guide: Writing an Effective CLAUDE.md

This is your reference for when you are asked to create or improve a `CLAUDE.md` (or `AGENTS.md`) for a project. Follow it as a working procedure, not as background reading. Everything you need is here.

## What you are actually building

A `CLAUDE.md` is the onboarding document for an agent that starts every session knowing nothing about the codebase. You are stateless between sessions: the only thing you know about a repo is what is in your context, and `CLAUDE.md` is the one file loaded into *every* session automatically. That makes it the highest-leverage file in the project — a single bad line degrades every future task, plan, and edit, not just one.

So your goal is narrow: write the smallest set of always-true facts and instructions that lets a capable but amnesiac agent do real work here. You are not writing documentation for humans, a style guide, or a feature wishlist.

## The mental model that drives every decision

Two facts govern everything you write:

1. **The instruction budget is small and shared.** A strong model reliably follows on the order of a couple hundred instructions; weaker or non-thinking models follow far fewer. The harness's own system prompt already consumes a large share before your file adds anything. Critically, when you exceed the budget, the model doesn't just drop the *last* rules — adherence to *all* of them degrades together. Every line you add taxes the lines you care about most.

2. **The harness is told your file might be noise.** `CLAUDE.md` is injected with a reminder that the context may not be relevant and should be ignored unless it clearly applies to the task at hand. The practical consequence: the more non-universal content you pack in, the more likely the agent disregards the *entire* file. Brevity and universality aren't aesthetic preferences — they are what keep the file from being ignored.

Therefore the prime directive: **include only what is both (a) universally relevant to the work done in this repo and (b) not already known or trivially discoverable.** Everything else goes elsewhere or gets cut.

## Cover three things, in your own words

A useful `CLAUDE.md` orients the agent on:

- **WHAT** — the stack and a map of the codebase. In a monorepo especially: what the apps are, what the shared packages are, and where to look for what. The agent can't navigate what it can't see a map of.
- **WHY** — the purpose of the project and the role each major part plays. This is the context that lets the agent make sane judgment calls.
- **HOW** — how to operate the repo: the package manager and runtime, how to build, how to run tests and typechecks, and how the agent verifies its own changes before claiming success.

Write these as plain, specific prose about *this* project. Do not pad them with generic explanations of well-known tools.

## What to include

- A short map of the repo and the purpose of its major parts.
- Project knowledge that wouldn't be in any model's training data: internal conventions, non-obvious architectural decisions, where specific kinds of data or config live.
- The exact commands to build, test, typecheck, lint, and verify — and which runner/package manager to use.
- Fixes for mistakes the agent *actually and repeatedly* makes in this repo. Add these as they're observed and remove them when they stop applying; this section should churn.
- Descriptive pointers to deeper docs the agent should read *when relevant* (see Progressive disclosure).

## What to leave out

- Common knowledge about the stack, or anything a capable model already knows.
- Anything the agent can learn in seconds by searching the codebase.
- Reading assignments for context it doesn't need yet.
- Code-style and formatting rules as prose (handle these deterministically — see below).
- One-off "hotfix" instructions that don't generalize. These are the main reason files bloat and then get ignored.
- Emphasis theater. Writing `CRITICAL` or `PRIORITY 0` does not buy attention; relevance and brevity do.

## Patterns that make instructions actually stick

**Phrase conditional rules as "If X, then Y."** Because the harness asks the model to apply context only when relevant, a trigger tells it exactly when a rule fires. Prefer `When touching the database layer, read docs/db-schema.md` over a free-floating `Always consult the schema docs`.

**Point, don't paste.** Never copy code snippets into `CLAUDE.md` or its referenced docs — they go stale the moment the code changes and silently mislead the agent. Use `path/to/file.ts:42` references to the authoritative source instead. Pointers age gracefully; copies rot.

**Progressive disclosure for everything non-universal.** You can't fit the whole project into a short file, so don't try. Move task-specific material into separate, self-descriptively named markdown files, and in `CLAUDE.md` list them with a one-line description each, instructing the agent to read the relevant ones before starting (or to propose which it wants to read). This keeps the always-on context lean while making depth retrievable on demand.

```
docs/
  building.md
  testing.md
  architecture.md
  db-schema.md
```

```markdown
## Reference docs — read when relevant
- Building & running locally → docs/building.md
- Test commands and conventions → docs/testing.md
- Service architecture & boundaries → docs/architecture.md
- When working with persistence → docs/db-schema.md
```

Caveat to plan around: agents follow these pointers inconsistently. For something that must happen on every relevant task, don't trust a pointer alone — back it with a hook or a command, or expect it may be missed.

**Use nested CLAUDE.md files for localized context.** A `CLAUDE.md` placed inside a subdirectory is pulled in automatically when the agent works on files there. Put a database-layer gotcha in `src/persistence/CLAUDE.md` so it loads only in that context — keeping substantial, focused notes near the code without inflating the root file's budget.

## Don't make the agent do a linter's job

Code-style rules in `CLAUDE.md` are the most common failure. The agent is slow, expensive, and non-deterministic compared to a formatter, and style prose drags irrelevant snippets into context, weakening everything else. Rely instead on:

- A formatter/linter that **auto-fixes** as much as is safe.
- A **Stop hook** that runs the linter/formatter and returns only the remaining errors for the agent to fix — so it isn't hunting for formatting problems itself.
- A **slash command** that points the agent at the diff or `git status` and applies guidelines as a separate step from implementation.

And trust in-context learning: given a few searches of existing code, the agent tends to match established patterns on its own. Let the codebase teach style; reserve `CLAUDE.md` for what the codebase can't.

## Write it by hand, then prune

Don't autogenerate the file or lean on `/init`. Because it touches every session, it's worth deliberate authorship. Autogenerated files run long, restate stack basics, and read poorly to both humans and the model. Start with a couple of paragraphs describing the codebase at a high level, add the build/test/verify commands, and then grow the rest only as you watch the agent stumble — adding a targeted fix each time and deleting it when obsolete. Keep the file short; if it's drifting past a couple hundred lines, that's the signal to split content into `docs/` or nested files, not to keep appending.

## A skeleton to start from

```markdown
# <Project name>

## What this is
<1–3 sentences: purpose of the project and who/what it serves.>

## Map
<Top-level layout and what each major dir/package is for. Heavier in monorepos.>

## How to work here
- Runtime / package manager: <e.g. bun, not node>
- Build: <command>
- Test: <command>
- Typecheck: <command>
- Verify a change is done: <what "green" means here>

## Project-specific gotchas
- <Non-obvious things the agent can't infer from the code or training data.>
- <If X, then Y conditional rules.>

## Reference docs — read when relevant
- <topic> → docs/<file>.md
- <topic> → docs/<file>.md
```

## Self-check before you finish

1. Does every line help with WHAT, WHY, or HOW for *this* repo?
2. Is each line universally relevant to the tasks actually run here? If not, move it to `docs/` or a nested file.
3. Could the agent get this from common knowledge or a quick codebase search? If yes, cut it.
4. Are style/formatting concerns delegated to a linter, formatter, or hook rather than written as prose?
5. Are deep-context docs *referenced* (ideally with an "if X" trigger) instead of pasted, with `file:line` pointers over copied code?
6. Is the file short — comfortably under a couple hundred lines, shorter if you can?
7. Did you author it deliberately rather than autogenerate it, and is the "gotchas" section something you'll keep pruning?