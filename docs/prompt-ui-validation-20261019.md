# Promptファイル選択と複数レコード学習 — 2026-10-19

新しい prompts/ にMOSS向けの5セット・各6件を追加し、V3ノード **MOSS Slider Train from Prompts** の prompt_yaml から選べるようにした。CLIP接続一つでファイル内の全target/positive/negativeをnative MOSS encoderへ渡し、各stepで一件ずつ順番に学習する。

## 実装と互換性

- prompt_files.py: リポ内のYAML一覧、UTF-8/BOMとスカラーalias対応のSafeLoader、重複キーや不正schema/パスの拒否、一回の読み取りに基づく本文とSHA256のsnapshot。
- nodes.py: 既存の学習設定を使うYAML選択ノード。従来のconditioning接続ノードとjob実行・出力処理を共有する。
- training.py / job.py: 単一のrole辞書と、複数レコードのlistを受理する。全件を事前検査し、未使用レコードはCPUへ保持する。stepごとに一件を選択し、学習式・seed/noise/timeのサンプリングは維持。reportにrecord_countと各stepのrecord_index、manifestにprompt_sourceを保存する。
- 既存の手動conditioningノードとworkflowを保持。anchor、neutral、男女分離の保持損失は今回追加していない。

## 検証

- 変更前のREDを記録後、全pytest **167 passed**（外部SWIG依存の警告2件）。
- 構文検査、git diff --checkはPASS。
- 5ファイルの各6レコード、YAML複数行とalias、拒否条件、hash変更、独立した条件長、複数レコードの循環、manifestのsource保持、従来/新V3ノードを検証。
- 稼働中ComfyUI0.42.0のobject_infoで新ノードと5項目のCOMBOを確認し、preflight通過後に実行した。
- R9700 / torch2.12.0+rocm7.14.0 / moodyMossMixFreeDownload_v1025STEPSCFG3_int8 / qwen3.5_9b_int8_convrotで、女性加齢YAMLを6 step、rank2、256×256、rollout4、checkpointing有効、swap0で実行。
- 実行ID: 40063fc3-8a8c-4e04-b896-ba89f1f795b4。terminal statusはsuccess。6レコードの利用順序は [0,1,2,3,4,5]。
- 全stepのloss/gradientは有限。lossは 0.018185, 0.023983, 0.013089, 0.009139, 0.005415, 0.008099。LoRAとJSON manifestを保存し、prompt_sourceの6本文とファイルSHA256一致を確認。
- 保存LoRA: C:/Users/inott/ComfyUI-Shared/models/loras/moss_sliders/moss_prompt_ui_20261019.safetensors。

## 利用と証拠

[プロンプトの一覧と編集方法](../prompts/README.md)、[UI用workflow](../workflows/moss_slider_train_prompts.json)、[API用workflow](../workflows/moss_slider_train_prompts_api.json)。既存のmodel/text encoderが別名なら選択肢から置き換える。

[test-results/prompt-ui-20261019](../test-results/prompt-ui-20261019/) にRED/GREENログ、稼働中ノード定義、実行API・submit・terminal history、report/manifest、summary.jsonとserverログを保存。

独立したブラウザー操作環境は利用不可だったため、画面のクリック・描画やスクリーンショットによる確認は未実行。UIへ供給される実際のCOMBO定義とAPI実行は確認済み。ruff/mypyの未導入という前回の検証制約も継続する。プロンプト5セット全体の画質効果や、6 step学習の十分性は評価していない。
