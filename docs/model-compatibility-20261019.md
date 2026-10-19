# 指定モデルへの互換対応 — 2026-10-19

**moodyMossMixFreeDownload_v1025STEPSCFG3_int8.safetensors での学習・保存・再適用を実機確認。** 以前の [実機検証](live-validation-20261019.md) で見つかった checkpoint 形式の非互換を解消した。

## 実装

- checkpoint.py: ヘッダーの __metadata__._quantization_metadata.layers と既存の .comfy_quant tensor 設定を読み、同じ検査・逐次ロード経路へ正規化。既知Linear・全shape・dtype・row scale・ConvRot規約・重複や設定競合を検査する。
- 指定モデルの184層を凍結したINT8 bufferとして保持。追加16層のbiasを保持し、層ごとの局所dequantと入力勾配に対応。LoRA対象は42×4=168投影のまま。
- global metadata形式でAttention設定がない場合はnative既定を保持し、学習中だけPyTorch Attentionへ切り替える。旧形式の明示Attention設定の検査も維持。
- model_adapter.py: aimdo有効時のnative遅延初期化に影響されず、専用学習モデルにmeta parameterを登録するLinearを使用。manual_castの計算経路と既存ComfyUIのglobal設定は維持。

## 検証

ComfyUI0.42.0 / Python3.12.11 / torch2.12.0+rocm7.14.0 / comfy-kitchen0.2.39 / comfy-aimdo0.5.6、AMD Radeon AI PRO R9700で実施。指定 image_moss_image_t2i.json と同じ基底・Qwen3.5 text encoder・Flux2 VAEを使用した。

| 試験 | 結果 |
|---|---|
| 全pytest | 147 passed、外部SWIG依存の警告2件 |
| 構文・git diff --check | PASS |
| 実checkpointの形式検査 | SUPPORTED、184 INT8 Linear、562 tensors |
| rank2 / alpha2 / 256×256 / 1 step / checkpointing有効 / swap0 | PASS、loss0.112691、grad_norm0.027264、保存成功 |
| 同設定で10 step / swap2 | PASS、全stepで有限loss・gradient、保存成功 |
| 保存した1 step LoRAの強度-1/0/+1、512×512・12推論step | 全3条件PASS |
| 強度0と未適用の比較 | 全画素一致 |
| 強度-1/+1と未適用の比較 | 平均絶対pixel差 1.2211 / 0.9671（0〜255尺度） |
| 保存した10 step LoRA | 756 tensors、全有限。252論理targetのdown行列が1 step後から更新され、252個のup行列が非ゼロ |
| 基底ファイルのSHA256再検査 | 学習前manifestと学習後の全ファイルhashが一致 |

基底SHA256: 68bc68c1c30ed7c222d721493629b2a42ec31893071aa395549001149b6fdfcf。

1 stepジョブ全体は26.11秒、10 stepジョブ全体は35.18秒。encode、全ファイルhash、モデルload、保存等を含む単発測定であり、optimizer1 stepあたりの時間ではない。stepごとにnoise/timeが異なるため、10件のlossを収束曲線とは解釈しない。

## 利用できるLoRAと再現用入力

最初の登録済みLoRAディレクトリ内の moss_sliders/ に保存済み。Applyの選択肢に表示される。

- moss_compat_20261019_1step.safetensors（再適用確認済み）
- moss_compat_20261019_10step_swap2.safetensors

保存ディレクトリ: C:/Users/inott/ComfyUI-Shared/models/loras/moss_sliders/。同名のmanifest JSONも保持する。

[test-results/live-compat-20261019](../test-results/live-compat-20261019/) に実行API JSON、submit・terminal history、manifest・学習report、画像、hash・tensor検査、RED/GREENログを保存。

- [1 step学習API](../test-results/live-compat-20261019/train-1step-api.json)
- [10 step学習API](../test-results/live-compat-20261019/train-10step-api.json)
- [Apply比較API](../test-results/live-compat-20261019/apply-comparison-api.json)
- [機械可読結果](../test-results/live-compat-20261019/summary.json)
- [未適用](../test-results/live-compat-20261019/apply-base.png) / [強度0](../test-results/live-compat-20261019/apply-zero.png) / [強度-1](../test-results/live-compat-20261019/apply-minus1.png) / [強度+1](../test-results/live-compat-20261019/apply-plus1.png)

## 限界と検証環境

短時間の小解像度学習と数値的な適用互換を確認した。100 step以上、実用的な年齢方向効果、複数prompt/seed、過学習、最低必要VRAM・peak VRAM、長時間安定性、checkpointing無効の実機試験、cancel/OOM復旧、フロントエンド画面操作は未評価。1 step画像のpixel差は意図した意味方向の効果の証明ではない。

ruff/mypyは現在の実行環境に未導入で、実行時に module unavailable となった。構文・差分検査と全pytestは通過。native reviewerも設定モデルgpt-5.4がこのアカウントで利用できず、独立レビューは完了していない。変更差分は直接レビューした。

pytestは C:/ComfyUI/.venv/Scripts/python.exe を使用し、既存Python312環境からpytestを追加のsys.pathで読み込んだ。PyTorch・comfy-kitchen・comfy-aimdoはComfyUI venvの版。ComfyUI rootを MOSS_COMFYUI_ROOT、TORCHINDUCTOR_CACHE_DIRをリポ内test-results/torch-cacheへ指定した。ComfyUI起動引数を維持するManager再起動APIでbackendを2回再読込し、実機試験は更新後のプロセスで実施した。途中の遅延初期化エラーと、WindowsのLoRA選択肢でパス区切りが異なった初回投入もhistoryに残している。最終成功はsummaryのprompt_idを参照。
