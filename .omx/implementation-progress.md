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

## Live verification — 2026-10-19

- Running Desktop ComfyUI 0.42.0 / torch 2.12.0+rocm7.14.0 / Radeon AI PRO R9700 probed at localhost:8000.
- User-selected image_moss_image_t2i.json uses moodyMossMixFreeDownload_v1025STEPSCFG3_int8.safetensors, not the original plan checkpoint.
- Baseline 512x512/12 steps PASS; after Train rejection, original workflow 1024x1024/25 steps including TextGenerate PASS.
- Inspect INVALID and Train ERROR at job.py:50 before model allocation: this checkpoint stores global _quantization_metadata and quantizes 184 linears, including 16 outside the fixed 168 adapter targets; per-tensor comfy_quant and attention records are absent.
- P8 real-model training remains BLOCKED_BY_CHECKPOINT_FORMAT. GPU forward/backward, export/apply, quality and training VRAM NOT_RUN. No product code or provided workflow changes.
- Evidence and implementation requirements: docs/live-validation-20261019.md; test-results/live-20261019/summary.json and terminal histories.

## Current model compatibility — 2026-10-19

- Resolved the preceding checkpoint-format blocker: global StarNodes metadata and184 frozen INT8 linears supported,168 adapter targets retained. Quantized non-target bias, metadata conflicts/schema, legacy format and aimdo lazy-init covered by regression tests.
- RED reproduced metadata rejection and aimdo-enabled missing Linear registration; GREEN147 tests passed (2 upstream SWIG warnings), syntax/diff checks PASS. ruff/mypy unavailable in the current environment. Independent native review unavailable because the configured role model is unsupported; direct diff review completed.
- Actual R9700/ROCm running ComfyUI: rank2/256px/1step swap0 and10step swap2 PASS with finite loss/gradients and LoRA publication. Apply-1/0/+1 PASS;0 equals baseline pixel-for-pixel;±1 differs.756 finite exported tensors;252 logical down matrices updated beyond step1. Base whole-file SHA256 unchanged.
- Current GPU smoke compatibility is PASS. P8 long-training quality, peak/min VRAM, cancel/OOM, frontend and broader settings remain unmeasured. Prior INVALID/blocked notes are historical.
- Evidence: docs/model-compatibility-20261019.md and test-results/live-compat-20261019/summary.json.

## Prompt YAML UI and multi-record training — 2026-10-19

- Added prompts/:5 MOSS English-caption sets,6 records each; aging/deaging by gender and adult breast size. Adapted directions and framing from Krea2 prompts using local MOSS USERGUIDE/caption guidance.
- Added MossSliderTrainFromPrompts with native CLIP and prompt_yaml COMBO. Retained manual-conditioning node; shared runner. Safe YAML snapshots include all texts and SHA256 in manifest. Unknown neutral/anchor fields are rejected rather than silently ignored.
- Extended job/trainer to validate every record and cycle one per optimizer step, preserving noise/time RNG and objective. Inactive records remain CPU-resident.
- RED observed; GREEN167 tests; syntax/diff checks PASS. Actual running object_info exposes5 filenames. R9700 actual encode→6-record/6-step training→LoRA export PASS, record order0..5, finite losses/gradients, source hash matched.
- No enabled browser surface for visual dropdown interaction; actual metadata/API verified. Prompt quality and all-set GPU effect evaluation remain unmeasured; ruff/mypy remain unavailable.
- Evidence: docs/prompt-ui-validation-20261019.md, workflows/moss_slider_train_prompts.json and test-results/prompt-ui-20261019/.

## Training progress/time visibility — 2026-10-19

- Added per-step perf_counter wall time with CUDA/ROCm synchronization, elapsed/ETA and total training-loop time in report/manifest. Seed/model numerical behavior remains covered independently of nondeterministic timings.
- Added Krea2-style INFO logs for every optimizer step plus encode/hash/load/train/save phases. Training ProgressBar no longer receives load/save totals; it remains at final N/N after saving.
- RED regression reproduced; GREEN169 tests, syntax/diff checks PASS. R9700 actual6-record/6-step train and export PASS. All6 per-step timing lines observed, positive durations in report+manifest, final ETA0.
- Updated running backend while queue was idle. Evidence: docs/training-progress-20261019.md, test-results/training-progress-20261019/step-progress.log and summary.json.
