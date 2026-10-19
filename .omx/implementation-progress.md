# Implementation ledger — plan: .omx/plans/2026-10-19-moss-slider-implementation-plan.md

- Scope: implement P1–P7; P8 GPU/real-checkpoint training remains outside this environment.
- Workspace: new, separate project directory, not a Git repository. Source references remain read-only; no worktree or commits are required by this plan.
- Pre-flight: checkpoint loader produces frozen native MOSS; adapter exposes raw velocity; LoRA owns only A/B; trainer consumes role-keyed normalized tensors; job owns resources and export.
- Parallel ownership: quantization; LoRA/export; trainer are separate native executor slices. Parent owns checkpoint/adapter/job/nodes/integration/docs.
- Environment: user updated comfy-aimdo to 0.5.6 and comfy-kitchen to 0.2.39. A project-local .test-deps fallback was installed before the user update; it is ignored and is not a product dependency.
- P1/P3 RED: checkpoint and conditioning tests written before implementation; pending initial run.
- P1/P3 initial GREEN: 13 checkpoint/conditioning tests; native model output/input-gradient/checkpointing/time/cleanup verified with a two-block CPU fixture.
- P2 complete: 32 tests, including FP64 gradcheck, independent/Comfy Kitchen ConvRot reference, saved-tensor memory contract.
- P4 complete: packed LoRA, logical export/import, separate hidden save reservation; first owned suite 26 tests passed.
- P5 complete: single-direction raw velocity teacher/student, frozen-base optimizer, stepwise seeded noise and time sampling; 31 targeted tests passed.
- Review fixes: full metadata schema (unit negative fixtures + native meta schema equality); fractional alpha FP32 canonicalization; LoRA/manifest filename+digest binding; safetensors rename as commit point. Each added regression was observed RED before its fix.
- Ruling: native float modules use `comfy.ops.manual_cast` (a disable_weight_init subclass), preserving F32 stored parameters while matching BF16 activation compute. A CPU BF16 regression reproduced the original dtype failure. Architecture and INT8 FrozenLinear remain unchanged; risk/cost is native cast-helper compatibility, covered by integration.
- Ruling: project has no Git baseline; verification and changed-file review replace commit-range scripts. No reference checkout is modified.

- Final GREEN: bare pytest 133 passed; validation runner syntax/ruff/mypy/unit115/integration18/CPU boundary/source integrity all PASS.
- Real checkpoint metadata-only inspection: SUPPORTED, 757 tensors, 168 INT8 projections, 42 attention configs. No real base matrices loaded.
- Independent review: 3 HIGH + 2 MEDIUM findings addressed with regression tests: complete schema, stochastic sampling, manifest binding, fractional alpha, final publication ordering. No deferred findings.
- P7 complete. P8 GPU/real-model training/quality/VRAM and live frontend QA remain NOT_RUN. UI/API workflow structural consistency PASS.
