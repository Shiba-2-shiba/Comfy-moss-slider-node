# Comfy MOSS Slider Node

MOSS-ImageのConvRot INT8モデルを凍結し、テキストの方向を制御するSlider LoRAを学習するComfyUI V3ノードです。INT8の基底とscaleを保持し、各Linearの計算時だけ浮動小数点へ戻して、FP32 LoRA A/Bのみを更新します。

現在は実験版です。CPUの小型native MOSS検証に加え、Radeon AI PRO R9700上で `moodyMossMixFreeDownload_v1025STEPSCFG3_int8.safetensors` のrank2・1/10 step学習、保存、強度-1/0/+1での推論を確認しています。長時間学習、必要VRAMの下限、slider効果の品質は未評価です。[実機互換検証](docs/model-compatibility-20261019.md)。

2026-10-19の互換対応後: **169 tests passed**、構文・差分チェックPASS。ruff/mypyは現在の環境に未導入のため今回未実行。初期実装時の133件・静的解析結果は[以前の検証概要](docs/validation.json)を参照してください。

2026-10-20のneutral L2正規化対応: **CPU 213 tests passed**、ruff・mypy・構文検査PASS。既定は`neutral_l2`、同梱全36レコードにneutralを追加しました。今回のGPU学習・画質評価は未実施です。[CPU検証レポート](test-results/neutral-l2/cpu-validation.json)。

## 必要な環境

- Python 3.10以上、PyTorch 2.9以上、safetensors 0.7以上。
- ネイティブMOSS、packed QKV、V3、LoRA bypass対応のComfyUI。開発基準はcommit `8f04fd6ecf4a548937971e840af2a985529fab76`です。
- 開発基準のComfyUI依存: `comfy-kitchen==0.2.39`、`comfy-aimdo==0.5.6`。ComfyUIを実行するPython環境の依存を合わせてください。
- 基底: `moss_image_sft_int8_convrot.safetensors` のtensor単位設定と、StarNodesのヘッダー `_quantization_metadata` 形式に対応します。後者は `moodyMossMixFreeDownload_v1025STEPSCFG3_int8.safetensors` の184個のINT8 Linearで検証しました。42 blockのpacked QKV、ConvRot group size 256を必要とします。追加のtext refiner・time・modulation・出力投影も凍結INT8で読み込み、LoRAの対象は従来の168投影です。BF16モデルを自動でINT8化する機能はありません。
- 実promptにはMOSS用Qwen3.5-9B text encoderとtokenizerが必要です。VAEは学習後の画像確認に必要で、テキストからのnoise rollout学習自体には使いません。

このフォルダーを`ComfyUI/custom_nodes/Comfy-moss-slider-node`として配置し、ComfyUIを再起動します。隣のMOSS-Image・sd-scripts・Krea2リポジトリは実行時依存ではありません。モデルはComfyUIの`diffusion_models`、text encoderは`text_encoders`の登録済み検索先に置きます。既存の`extra_model_paths.yaml`による検索先も利用できます。

## ノードと接続

カテゴリーは`MOSS Image/Slider LoRA`です。

| ノード | 用途 |
|---|---|
| MOSS Slider Inspect INT8 | 大きい重みをロードせず、header・量子化形式・形状を検査 |
| MOSS Slider Train INT8 | target/positive/negative条件から単方向sliderを学習し、LoRAとreportを出力 |
| MOSS Slider Train from Prompts | MOSS CLIPとprompts/のYAMLを選び、全レコードを順番に学習 |
| MOSS Slider Apply INT8 | LoRAのQKVを結合してbypass適用し、INT8基底をmerge・書換せずMODELを返す |

プロンプト一覧から選ぶ場合は、[MOSS Slider Train from Promptsのworkflow](workflows/moss_slider_train_prompts.json)を開き、標準のLoad CLIP（type=moss_image）を接続して **prompt_yaml** を選択します。[prompts/](prompts/README.md)に加齢・若返りの男女別と成人女性の胸サイズのアニメ調5セットと、実写調の女性加齢1セット（各6件）を同梱しています。stepsは更新総数で、全レコードを一件ずつ循環して使います。選択ファイルの本文とSHA256はmanifestに保存します。

ノードの既定値と同梱学習ワークフローは、提供された `deaging_slider.json` の学習設定に合わせ、600 steps・rank16・alpha2・学習率0.0002・eta1.5・768×768・rollout_steps=16へ揃えています。seed=0、gradient_checkpointing=true、blocks_to_swap=0、teacher_normalization=neutral_l2です。プロンプト選択型の初期選択は今回使用された `deaging_slider_fullbody.yaml`、出力名は `deaging_slider` です。ワークフローの基底も今回と同じ `moodyMossMixFreeDownload_v1025STEPSCFG3_int8.safetensors` に揃えています。設定の根拠と確認範囲は[prompts/README.md](prompts/README.md)を参照してください。

手動でconditioningを接続する場合は従来のMOSS Slider Train INT8を使えます。

1. 標準の`Load CLIP`でMOSS用text encoderを読み、typeを`moss_image`にします。
2. 3個の`CLIP Text Encode`を作り、target/positive/negativeへ接続します。例: target=`portrait of an adult person`、positive=`portrait of an elderly adult person`、negative=`portrait of a young adult person`。
3. Trainの`checkpoint_name`へ指定INT8基底を選びます。まずrank 2・小解像度・1 stepの実機smokeから段階的に増やしてください。
4. 学習完了後、`output/moss_sliders/`のLoRAと同名JSONを、使用するLoRAフォルダーへ一緒にコピーし、ComfyUIのモデル一覧を更新します。推論用workflowでApplyの`lora_name`にコピー先の保存名を選び、標準`Load Diffusion Model`から**同じ基底**をMODELへ接続します。
5. ApplyのMODELを通常のMOSS samplerへ渡します。画像確認にはFlux2 VAEを使用します。

Trainはqueueごとに学習する出力ノードです。画像比較では保存済みLoRAをApplyで読み、学習用と推論用workflowを分けます。

`workflows/moss_slider_train.json`はComfyUIで開く編集用graph、`workflows/moss_slider_train_api.json`はAPI形式の同じ学習例です。model/text encoderのfilenameは自分の登録名へ置き換えます。

## 条件入力の対応範囲

各roleはentryが1個、batch 1、shape `[1,L,4096]`、token数1〜4096に限定しています。`attention_mask`があれば右paddingだけをtrimし、条件ごとの真の長さを使います。maskがなければ全tokenを有効とします。

複数entry、左padding・内部にholeのあるmask、全padding、NaN/Inf、schedule、領域条件、ControlNet、hooks、追加guidance等はエラーにします。`pooled_output`はMOSSでは使いません。通常のMOSS Text Encode出力をそのまま接続する構成が初期対象です。

## 学習中の進捗

ComfyUIのログに、プロンプトのエンコード件数、checkpointのhash計算、モデルロード、学習開始、各step、保存完了を表示します。学習中の標準進捗バーはoptimizerの更新数で進みます。

~~~text
MOSS Slider step 2/6 | 1.30s/step | elapsed 4.6s | ETA 9.2s | loss=0.0239832 grad=0.066817
~~~

秒/stepは、そのstepのrollout・teacher/student予測・backward・optimizer更新を含む実時間です。CUDA/ROCmではGPU処理の完了を待って計測します。elapsedとETAは学習ループの時間で、事前のエンコード・hash・ロードや最後の保存は含みません。ETAは完了済みstepの平均からの目安で、rolloutの長さなどで変動します。各stepのseconds・elapsed_seconds・eta_secondsと、学習ループのtotal_secondsはreport/manifestにも残ります。

現在のDesktop環境では、ComfyUIログのファイルは C:/ComfyUI/user/comfyui_8000.log です。[実機での進捗表示確認](docs/training-progress-20261019.md)。

## 学習設定

| 設定 | 内容 |
|---|---|
| `steps` | optimizer更新回数。既定600 |
| `rank` / `alpha` | LoRA rankと`alpha/rank`係数。既定16 / 2（係数0.125） |
| `learning_rate` | AdamWの学習率。既定0.0002、weight decayは0 |
| `eta` | teacherの`positive-negative`方向の強さ。既定1.5。samplerのCFG値とは別 |
| `width` / `height` | 学習解像度。既定768 / 768 |
| `teacher_normalization` | 既定`neutral_l2`。教師全体のL2ノルムをneutralに合わせる。`none`で従来の教師式を使用 |
| `rollout_steps` | native shift=5のEuler schedule。既定16 |
| `gradient_checkpointing` | block単位のnon-reentrant再計算。既定有効 |
| `blocks_to_swap` | 末尾N blockのI8/scaleをCPUに残し、Linearごとに同期転送。0〜42 |

teacherはLoRA無効の同じ量子化基底から作ります。まず`teacher=base(target)+eta*(base(positive)-base(negative))`を作り、既定の`neutral_l2`ではFP32・sample単位で`teacher *= ||base(neutral)|| / ||teacher||`とします。studentはtarget条件とLoRA強度+1で、raw flow velocityのMSEを学習します。Attentionはteacher/studentともPyTorchの勾配対応経路です。推論時のbackendとの差はモデル・環境別の検証対象です。

手動ノードには任意の`neutral_conditioning`、YAMLには任意の`neutral`を指定できます。省略時はtargetを参照し、同梱6ファイル・全36レコードには`neutral=target`を明示しています。targetと同じneutralでは教師予測を再利用し、追加forwardを行いません。異なるneutralを指定した場合は教師forwardが1回増えます。

**正規化の既定値は`neutral_l2`です。以前と同じ学習計算を再現する場合は`teacher_normalization=none`を指定してください。** 項目のない既存API・ワークフローも受理しますが、既定のneutral正規化が適用されます。保存済みLoRAにはこの変更は影響しません。

教師または参照のL2ノルムが`1e-8`以下なら、そのsampleは正規化をスキップします。倍率・参照元・skipの有無はJSONの各stepの`teacher_normalization_info`へ記録します。非有限値になった場合は学習を停止します。異なるneutralではeta=0でも教師全体が再スケールされ得ます。この処理はanchorによる保持lossではなく、服装保持や画像品質の改善は別途比較が必要です。`none`ではneutralの教師予測や正規化を実行しません。

内部は42×4=168 adapterです。QKVはAを共有し、保存時にAをq/k/vへ複製、Bを行方向に分割するため、252論理target・756 tensorの標準LoRAになります。独立したAを持つq/k/v LoRAを同rankへ圧縮して読み込むことはしません。text encoder、text refiner、norm、modulation、入出力投影は学習しません。

LoRA強度0は基底に一致し、負値も適用できます。単方向学習なので、負値で望む逆方向の画質効果が得られる保証はありません。複数slider、領域mask、bidirectional、画像dataset、resumeは初期版の対象外です。

## 保存と推論

出力はComfyUIに設定されたoutputディレクトリ内の`moss_sliders/`です。現在の環境では`C:\ComfyUI\output\moss_sliders\`になります。追加モデルパスやLoRA検索先の登録順には依存しません。保存先のフルパスはノードの表示、ログ、reportの`lora_path`と`manifest_path`で確認できます。`lora_name`出力はoutputからの相対パスです。Applyや標準LoRA loaderで使用するには、LoRAと同名JSONを登録済みLoRAフォルダーへ一緒にコピーし、モデル一覧を更新してください。同名ファイルは上書きせずsuffixを付けます。`.safetensors`と同名の`.json`を同じ名前のまま一緒に保持してください。manifestには基底全体とLoRA本体のSHA256、設定、実行版、loss、使用した時刻等を記録します。JSONを配置してから完成したLoRAを最終名へrenameするcommit方式です。プロセスが途中終了しても、JSONなしの未完成LoRAを最終名で公開しません。孤立JSON等が残った場合も再実行は既存名を上書きしません。

専用Applyは標準`Load Diffusion Model`由来のMODELから基底を識別し、manifestのSHA256と照合します。別基底・既存LoRAでpatch済みのMODEL・manifest欠落を拒否します。QKVをpackedキーへ戻してComfyUIのbypassを使い、weight mergeや再量子化を行いません。

保存LoRAの論理キーは標準LoRA loaderにも対応します。ただし、標準loaderのmerge/requantizeは専用Applyの「INT8基底を保持する」経路と同じ数値になるとは限らず、標準loaderはこのmanifestの基底一致を強制検証しません。

## CPU検証

開発用のpytestがある環境で、プロジェクト直下から実行します。実ComfyUIを明示し、CPU設定をimport前に適用します。

```powershell
python tools/inspect_checkpoint.py --checkpoint "../moss_image_sft_int8_convrot.safetensors" --metadata-only
python tools/run_cpu_validation.py --comfyui-root "../ComfyUI" --report "test-results/cpu-validation.json"
python -m pytest -m "not gpu" -q
```

検査CLIは基底行列を読みません。全ファイルhashを求める場合だけ`--full-hash`を指定します。学習jobの実行時にはmanifest用に全hashを計算します。

CPU suiteでは、小型native MOSSへの逐次ロード、ConvRot出力・input gradient、FP64 gradcheck、早い層のLoRA勾配、checkpointing、全42層の標準key mapping、±/0 bypass、基底不変、条件入力、atomic export、cancel後の復旧、V3 metadataを検証します。巨大なMOSS実モデルでのGPU forward/backwardは含めません。

`test-results/cpu-validation.json`と各stageログが検証結果です。ComfyUI依存不足等はintegrationを`BLOCKED`として記録し、mockによる成功に置き換えません。

## 実機へ移るとき

指定基底でLoRAなしのbaselineを保存し、rank2・1 stepから始めます。その後10/100 step、checkpointing/staging、cancel/retry、peak VRAM/RAMとstep時間を確認します。画像は同じseedで強度-2/-1/0/+1/+2と未見promptを比較してください。CPUでの数値整合から必要VRAMや生成品質は判定できません。

実装の根拠と段階別条件は[実装計画書](.omx/plans/2026-10-19-moss-slider-implementation-plan.md)、実装・検証記録は[進捗記録](.omx/implementation-progress.md)を参照してください。
