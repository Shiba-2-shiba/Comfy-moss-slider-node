# SDD ledger — plan: .omx/plans/2026-10-20-neutral-l2-normalization-implementation-plan.md

Ruling: User explicitly requests neutral as the default. Use neutral_l2 in config, nodes, omitted API fields, and bundled workflows; none remains the explicit legacy path. This overrides the plan's original default-none requirement.
Ruling: Execute in the user's named checkout, preserving their untracked plans. No worktree, branch switch, or commits are needed to deliver the requested reviewable changes.
Pre-flight: Task 1 math/config feeds Task 2 training and Task 3 UI; mode names and defaults must match. Task 4 verifies the combined interfaces.
Task 1: in progress — captured legacy fixed CPU toy baseline at test-results/neutral-l2/baseline-before.pt before production edits.
Ruling: User additionally requests neutral in existing presets. Added explicit neutral=target to all 36 records; independently compared all original target/positive/negative strings before/after. Prompt hashes intentionally change.
Task 1: complete — numeric primitive/config and degenerate tests RED→GREEN.
Task 2: complete — optional roles, target fallback, normalized teachers and JSON diagnostics pass targeted tests. Exact equal neutral=target reuses target prediction (CPU equality), preserving preset training cost.
Task 3: in progress — schemas, defaults and workflows updated; checking actual Comfy prompt validation and documentation.
Task 3: complete — both V3 schemas and real Comfy prompt validation accept omitted optional inputs; default neutral_l2 and workflow widget order verified.
Task 4: complete — 174 unit +39 integration tests passed, no skips; syntax, ruff, mypy, CPU boundary and source integrity passed. Full report: test-results/neutral-l2/cpu-validation.json.
Legacy none: before/after CPU toy weights and numerical step records are exactly equal. All36 original three prompt strings are unchanged.
Review: independent code-reviewer approved with zero findings. GPU/UI-browser execution not performed; CPU-only torch environment. Image-quality improvement remains unmeasured.
No commits, merges, pushes or external services were changed.
