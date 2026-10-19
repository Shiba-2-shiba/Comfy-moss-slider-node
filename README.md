# Comfy MOSS Slider Node

MOSS-ImageのConvRot INT8モデルを凍結し、テキストの方向を制御するSlider LoRAを学習するComfyUI V3ノードです。INT8の基底とscaleを保持し、各Linearの計算時だけ浮動小数点へ戻して、FP32 LoRA A/Bのみを更新します。

現在は**CPUの小型native MOSS構造まで検証した実験版**です。指定された実checkpointでのGPU学習、必要VRAM、速度、生成品質は未検証です。

2026-10-19の開発検証: **133 tests passed**、ruff・コアmypy・構文検査PASS。[検証概要](docs/validation.json)。

## 必要な環境

- Python 3.10以上、PyTorch 2.9以上、safetensors 0.7以上。
- ネイティブMOSS、packed QKV、V3、LoRA bypass対応のComfyUI。開発基準はcommit `8f04fd6ecf4a548937971e840af2a985529fab76`です。
- 開発基準のComfyUI依存: `comfy-kitchen==0.2.39`、`comfy-aimdo==0.5.6`。ComfyUIを実行するPython環境の依存を合わせてください。
- 基底: `moss_image_sft_int8_convrot.safetensors`。42 blockのpacked QKV、168個のINT8 Linear、group size 256に対応します。BF16モデルを自動でINT8化する機能はありません。
- 実promptにはMOSS用Qwen3.5-9B text encoderとtokenizerが必要です。VAEは学習後の画像確認に必要で、テキストからのnoise rollout学習自体には使いません。

このフォルダーを`ComfyUI/custom_nodes/Comfy-moss-slider-node`として配置し、ComfyUIを再起動します。隣のMOSS-Image・sd-scripts・Krea2リポジトリは実行時依存ではありません。モデルはComfyUIの`diffusion_models`、text encoderは`text_encoders`の登録済み検索先に置きます。既存の`extra_model_paths.yaml`による検索先も利用できます。

## ノードと接続

カテゴリーは`MOSS Image/Slider LoRA`です。

| ノード | 用途 |
|---|---|
| MOSS Slider Inspect INT8 | 大きい重みをロードせず、header・量子化形式・形状を検査 |
| MOSS Slider Train INT8 | target/positive/negative条件から単方向sliderを学習し、LoRAとreportを出力 |
| MOSS Slider Apply INT8 | LoRAのQKVを結合してbypass適用し、INT8基底をmerge・書換せずMODELを返す |

1. 標準の`Load CLIP`でMOSS用text encoderを読み、typeを`moss_image`にします。
2. 3個の`CLIP Text Encode`を作り、target/positive/negativeへ接続します。例: target=`portrait of an adult person`、positive=`portrait of an elderly adult person`、negative=`portrait of a young adult person`。
3. Trainの`checkpoint_name`へ指定INT8基底を選びます。まずrank 2・小解像度・1 stepの実機smokeから段階的に増やしてください。
4. 学習完了後、推論用workflowでApplyの`lora_name`に保存名を選び、標準`Load Diffusion Model`から**同じ基底**をMODELへ接続します。
5. ApplyのMODELを通常のMOSS samplerへ渡します。画像確認にはFlux2 VAEを使用します。

Trainはqueueごとに学習する出力ノードです。画像比較では保存済みLoRAをApplyで読み、学習用と推論用workflowを分けます。

`workflows/moss_slider_train.json`はComfyUIで開く編集用graph、`workflows/moss_slider_train_api.json`はAPI形式の同じ学習例です。model/text encoderのfilenameは自分の登録名へ置き換えます。

## 条件入力の対応範囲

各roleはentryが1個、batch 1、shape `[1,L,4096]`、token数1〜4096に限定しています。`attention_mask`があれば右paddingだけをtrimし、条件ごとの真の長さを使います。maskがなければ全tokenを有効とします。

複数entry、左padding・内部にholeのあるmask、全padding、NaN/Inf、schedule、領域条件、ControlNet、hooks、追加guidance等はエラーにします。`pooled_output`はMOSSでは使いません。通常のMOSS Text Encode出力をそのまま接続する構成が初期対象です。

## 学習設定

| 設定 | 内容 |
|---|---|
| `steps` | optimizer更新回数。既定100 |
| `rank` / `alpha` | LoRA rankと`alpha/rank`係数。既定8 / 8 |
| `learning_rate` | AdamWの学習率。既定1e-4、weight decayは0 |
| `eta` | teacherの`positive-negative`方向の強さ。samplerのCFG値とは別 |
| `rollout_steps` | native shift=5のEuler schedule。既定8 |
| `gradient_checkpointing` | block単位のnon-reentrant再計算。既定有効 |
| `blocks_to_swap` | 末尾N blockのI8/scaleをCPUに残し、Linearごとに同期転送。0〜42 |

teacherはLoRA無効の同じ量子化基底から作ります。`teacher=base(target)+eta*(base(positive)-base(negative))`、studentはtarget条件とLoRA強度+1で、raw flow velocityのMSEを学習します。Attentionはteacher/studentともPyTorchの勾配対応経路です。推論時のINT8 Attentionとの差はGPU検証の対象です。

内部は42×4=168 adapterです。QKVはAを共有し、保存時にAをq/k/vへ複製、Bを行方向に分割するため、252論理target・756 tensorの標準LoRAになります。独立したAを持つq/k/v LoRAを同rankへ圧縮して読み込むことはしません。text encoder、text refiner、norm、modulation、入出力投影は学習しません。

LoRA強度0は基底に一致し、負値も適用できます。単方向学習なので、負値で望む逆方向の画質効果が得られる保証はありません。複数slider、領域mask、bidirectional、画像dataset、resumeは初期版の対象外です。

## 保存と推論

出力は最初の登録済みLoRAディレクトリ内の`moss_sliders/`です。同名ファイルは上書きせずsuffixを付けます。`.safetensors`と同名の`.json`を同じ名前のまま一緒に保持してください。manifestには基底全体とLoRA本体のSHA256、設定、実行版、loss、使用した時刻等を記録します。JSONを配置してから完成したLoRAを最終名へrenameするcommit方式です。プロセスが途中終了しても、JSONなしの未完成LoRAを最終名で公開しません。孤立JSON等が残った場合も再実行は既存名を上書きしません。

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
