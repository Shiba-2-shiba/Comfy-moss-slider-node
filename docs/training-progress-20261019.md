# 学習進捗とstep時間 — 2026-10-19

Krea2 SliderのLOGGERによるstep別の秒数表示を参考に、MOSSにも段階ログとstep時間を追加した。

- プロンプトエンコード: 完了レコード数 / 全レコード数。
- 準備: inference modelのunload、checkpoint全体のhash計算開始・完了。
- ロード: tensor読込の途中経過を約10%ごとに記録。
- 学習: 0/Nの開始表示、各stepの完了数、秒/step、経過秒、残り秒の目安、loss、grad_norm。
- 保存: 開始・公開完了。

以前はload/saveの総数も学習用ProgressBarへ渡していた。今回の変更では、準備時に0へ戻し、学習中は更新数だけを渡す。保存時に総数1へリセットしない。

## 時間の定義

各stepの最初からoptimizer更新とGPU完了までをperf_counterで計測する。CUDA/ROCmではtorch.cuda.synchronize(device)でGPUの非同期実行を含める。CPUでは同期APIを呼ばない。

secondsはそのstepの実時間、elapsed_secondsは学習ループ開始からの経過時間、eta_secondsはelapsed/completed_steps×残りstep数。total_secondsは学習ループ全体。事前のエンコード・hash・モデルロード、最後の保存・後処理は含まない。初回のwarmupやstepごとのrollout数で所要時間が異なるためETAは目安。

## 検証

- RED: step時間の欠落と、load/saveの総数が進捗バーへ混ざることをテストで再現。
- GREEN: 全pytest169件通過、外部SWIG依存の警告2件。固定時計で2/3/4秒、経過2/5/9秒、ETA4/2.5/0秒を検証。既存のseed再現性は時間フィールドを除いた数値とparameterで確認。
- 構文、git diff --checkはPASS。ruff/mypyは現在の環境に未導入という制約が継続。
- 実行キューが空の状態でManagerの既存引数を維持する再起動APIを実行し、更新後のbackendで検証。
- ComfyUI0.42.0 / R9700 / torch2.12.0+rocm7.14.0、既存のmoodyMoss INT8とMOSS CLIP、女性加齢6レコード、rank2・256×256・rollout4・6 stepで学習・LoRA保存成功。
- 実行ID: c48164ab-3286-44d4-8b7e-61d595bf2161。全6stepでログ出力を確認し、reportとmanifestに正の秒数、単調な経過時間、最終ETA0が保存された。全loss/gradientは有限。

実際のログ:

~~~text
MOSS Slider step 1/6 | 3.28s/step | elapsed 3.3s | ETA 16.4s | loss=0.0181845 grad=0.02564
MOSS Slider step 2/6 | 1.30s/step | elapsed 4.6s | ETA 9.2s | loss=0.0239832 grad=0.066817
MOSS Slider step 3/6 | 1.16s/step | elapsed 5.8s | ETA 5.8s | loss=0.0128296 grad=0.022772
MOSS Slider step 4/6 | 1.46s/step | elapsed 7.2s | ETA 3.6s | loss=0.00928685 grad=0.016036
MOSS Slider step 5/6 | 1.59s/step | elapsed 8.8s | ETA 1.8s | loss=0.00528439 grad=0.0058903
MOSS Slider step 6/6 | 1.59s/step | elapsed 10.4s | ETA 0.0s | loss=0.00804771 grad=0.014547
~~~

学習ループ全体は約10.39秒。これは単発の検証値で、一般的な性能目安ではない。

[test-results/training-progress-20261019](../test-results/training-progress-20261019/) にRED/GREENログ、実行APIとterminal history、step-progress.log、report/manifestとsummary.jsonを保存。ComfyUIのライブ画面の描画は未確認だが、実サーバーのログとreportの出力は確認済み。
