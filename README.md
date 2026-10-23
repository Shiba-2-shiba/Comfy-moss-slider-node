# Comfy MOSS Slider Node

MOSS-ImageのConvRot INT8モデルで、プロンプトに指定した変化をLoRAの強度で調整できる「Slider LoRA」を学習・適用するComfyUIカスタムノードです。

INT8の基底モデルを凍結したまま、FP32のLoRAパラメーターだけを学習します。専用のApplyノードでは基底モデルへ重みをマージせずに適用でき、複数のスライダーを個別の強度で組み合わせられます。

**実験版です。** 短時間のGPU学習・保存・再適用は確認していますが、モデルやプロンプトごとの品質、最低必要VRAM、長時間学習の安定性は保証していません。モデル・text encoder・VAE・学習済みLoRAは同梱していません。

## 主な機能

- ConvRot INT8チェックポイントの形式・形状を、モデル重み全体をロードせずに検査。
- YAMLのプロンプトセット、または手動で接続したconditioningからSlider LoRAを学習。
- 学習中のstep時間・loss・gradient・ETAをComfyUIのログに表示。
- LoRAと、基底モデルのSHA256・学習設定・プロンプト・実行結果を記録したJSONを保存。
- 基底モデルの一致を確認してLoRAを適用。強度0・負の強度・複数ノードの連結に対応。

## 必要な環境

| 項目 | 要件 |
|---|---|
| ComfyUI | ネイティブMOSS、packed QKV、V3 API、LoRA bypassに対応する版 |
| Python | 3.10以上 |
| PyTorch / safetensors | PyTorch 2.9以上 / safetensors 0.7以上 |
| PyYAML | プロンプトYAMLの読み込みに使用。ComfyUI環境のものを利用 |
| 基底モデル | 対応するMOSS-Image ConvRot INT8チェックポイント |
| Text encoder | MOSS用Qwen3.5-9B text encoderとtokenizer |
| VAE | 生成画像の確認にはFlux2 VAEを使用。学習自体には不要 |

開発・検証基準のComfyUIはcommit `8f04fd6ecf4a548937971e840af2a985529fab76`です。その環境では`comfy-kitchen==0.2.39`、`comfy-aimdo==0.5.6`を使用しました。これらは検証時の構成であり、他の版での動作を保証するものではありません。

### 対応チェックポイント

- 層ごとの`.comfy_quant` tensor設定、またはヘッダーの`_quantization_metadata.layers`設定を持つConvRot INT8モデル。
- ComfyUI本体と同様に層ごとの設定を読み、トップ階層の`format`・`format_version`・変換ツール名では判定しません。両方の設定がある層ではヘッダー側を優先します。
- ConvRot設定は層の直下、または`params`内から読み取ります。`convrot_groupsize`省略時は256、Attention設定省略時はnative既定を使用します。

`moss_image_sft_int8_convrot.safetensors`および`moodyMossMixFreeDownload_v1025STEPSCFG3_int8.safetensors`で実機検証済みです。

対応可否はモデル名ではなく、各層の量子化設定と実際の重み・スケールの形状で判定します。同じMOSS-Image構造の派生モデルであれば、metadataの表現が上記の範囲で異なっていても、モデル名ごとの追加処理なしで利用できる設計です。

| 項目 | 対応条件 |
|---|---|
| モデル構造 | 42 blockのpacked QKV。各パラメーターの形状が既存のnative MOSS-Image構造と一致すること |
| LoRA対象の基底重み | `int8_tensorwise`形式のINT8重みで、ConvRotが有効であること |
| ConvRot group size | 256。省略時も256として解釈 |
| 量子化スケール | 各出力行に対応するFP32スケール。形状は`[out_features, 1]` |

他の量子化方式、異なるモデル構造、BF16モデルの自動INT8化には対応していません。

初めて使うモデルは、まず**MOSS Slider Inspect INT8**で`SUPPORTED`になることを確認してください。これは構造・形式の検査に合格したことを示します。続いてrank 2・小解像度・1 stepで学習・保存・適用を確認してから、学習設定を増やしてください。他モデルの長時間学習や品質は、モデル・プロンプトごとの確認が必要です。

## インストール

1. ComfyUIの`custom_nodes`ディレクトリで、このリポジトリをcloneします。

   ```sh
   git clone https://github.com/Shiba-2-shiba/Comfy-moss-slider-node.git
   ```

2. ComfyUIを実行するPython環境に、上記の依存関係があることを確認します。
3. 必要なモデルを、ComfyUIに登録された検索先へ配置します。

   | ファイル | 標準の配置先 |
   |---|---|
   | ConvRot INT8基底モデル | `ComfyUI/models/diffusion_models/` |
   | MOSS用text encoder | `ComfyUI/models/text_encoders/` |
   | Flux2 VAE | `ComfyUI/models/vae/` |

4. ComfyUIを再起動します。ノードは`MOSS Image/Slider LoRA`カテゴリーに表示されます。

`extra_model_paths.yaml`に登録した検索先も利用できます。MOSS-Image・sd-scripts・Krea2の別リポジトリを隣に配置する必要はありません。

## ノード一覧

| ノード | 用途 |
|---|---|
| MOSS Slider Inspect INT8 | チェックポイントのヘッダー・量子化形式・形状を検査 |
| MOSS Slider Train from Prompts | MOSS用CLIPとYAMLプロンプトを使って学習 |
| MOSS Slider Train INT8 | target / positive / negativeのconditioningを手動接続して学習 |
| MOSS Slider Apply INT8 | 保存済みLoRAをINT8基底へbypass適用し、MODELを出力 |

## 使い方

### 1. プロンプトセットから学習する

1. [プロンプト選択型ワークフロー](workflows/moss_slider_train_prompts.json)をComfyUIで開きます。
2. 標準の**Load CLIP**でMOSS用text encoderを選び、`type=moss_image`にします。
3. Trainノードの`checkpoint_name`で対応するINT8基底を選びます。
4. `prompt_yaml`で学習するプロンプトセットを選び、学習設定と`output_prefix`を確認します。
5. ワークフローを実行します。完了すると、LoRAと同名のJSONが`output/moss_sliders/`へ保存されます。

ワークフロー内のモデル名は例です。手元の登録名へ変更してください。初期選択は`deaging_slider_fullbody.yaml`、出力名は`moss_slider`です。

初回はrank 2・小解像度・1 stepで、学習から保存・適用まで動くことを確認してから設定を増やしてください。同梱ワークフローの600 stepsは動作確認用の最小設定ではありません。

[prompts/](prompts/)には13セット・合計90レコードを同梱しています。初音ミク外見を含む11セットは各6レコード、実写調の女性限定若返りv2と実写胸サイズは各12レコードです。

| セット | 内容 |
|---|---|
| `aging_slider_fullbody.yaml` | アニメ調・成人女性の皮膚の加齢 |
| `aging_slider_fullbody_male.yaml` | アニメ調・成人男性の皮膚の加齢 |
| `aging_slider_realistic.yaml` | 実写調・成人女性の加齢 |
| `aging_slider_realistic_male.yaml` | 実写調・成人男性の皮膚の加齢（6レコード、実画像未検証） |
| `deaging_slider_fullbody.yaml` | アニメ調・着衣の成人女性から幼児への年齢・体格変化 |
| `deaging_slider_fullbody_male.yaml` | アニメ調・着衣の成人男性から幼児への年齢・体格変化 |
| `deaging_slider_realistic.yaml` | 実写調・着衣の成人女性1人から幼児への年齢・体格変化（6レコード、実画像未検証） |
| `deaging_slider_realistic_male.yaml` | 実写調・着衣の成人男性1人から幼児への年齢・体格変化（6レコード、実画像未検証） |
| `breast_size_slider.yaml` | アニメ調・着衣の成人女性の胸サイズ変化 |
| `breast_size_slider_realistic.yaml` | 実写調・着衣の成人女性の胸サイズ変化。顔・髪・衣装を変えた12レコード（実画像未検証） |
| `hatsune_miku_identity_no_costume.yaml` | 初音ミク風の髪・瞳・顔立ちを直接記述。6種類の普段着、腰上中心、肯定形の6レコード（修正版の生成・学習効果は未検証） |
| `hatsune_miku_identity_no_costume_realistic.yaml` | 実写の成人女性を初音ミク風の外見へ。キャラ名や装備の列挙を使わず、普段着を対内で固定（6レコード、修正版の生成・学習効果は未検証） |
| `realistic_female_only_deaging_slider_v2.yaml` | 実写調・男女2人の全身構図で、女性だけを成人から幼児へ変えることを狙う評価用セット（12レコード、MOSS実画像未検証） |

実写で胸サイズを評価する場合は`breast_size_slider_realistic.yaml`を選び、新しい出力名で再学習してください。既存の`breast_size_slider.yaml`はアニメ調です。詳しくは[実写胸サイズの評価](prompts/README.md#実写胸サイズの評価)を参照してください。

MoodyMossなどの実写寄りモデルでは、加齢・若返り・初音ミク外見にも実写調のセットを選べます。[実写調セットの選び方](prompts/README.md#moodymossなどの実写寄りモデルで使うセット)を参照してください。

`steps`はファイル全体でのoptimizer更新総数です。1更新につき1レコードを順番に使うため、6レコードで600 stepsなら各レコードを100回、12レコードなら各50回使います。新しい実写調セットの[試し方と評価項目](prompts/README.md#女性限定若返りv2の評価)も参照してください。

独自セットは`prompts/`にUTF-8の`.yaml`または`.yml`として保存し、ComfyUIを再起動すると選択できます。基本形式は次のとおりです。

```yaml
- target: A portrait of an adult person with smooth facial skin.
  positive: A portrait of an adult person with fine facial wrinkles.
  negative: A portrait of an adult person with smooth facial skin.
```

`target`は学習する基準、`positive`は強度+1で向かわせたい特徴、`negative`は弱めたい特徴です。`negative`は学習方向の比較用キャプションです。任意の`neutral`で正規化の参照を指定でき、省略時は`target`を使います。1ファイル1〜64レコード、2 MiBまで対応します。詳細は[プロンプトの形式](prompts/README.md)を参照してください。

### 2. 手動でconditioningを接続する

[手動接続型ワークフロー](workflows/moss_slider_train.json)を開き、MOSS用の**Load CLIP**と3個の**CLIP Text Encode**を用意して、Trainの`target_conditioning`・`positive_conditioning`・`negative_conditioning`へ接続します。

各入力はentry 1個・batch 1・shape `[1, L, 4096]`・1〜4096 tokenのMOSS conditioningに対応します。右paddingはmaskに基づいて除去します。複数entry、左padding、schedule、領域条件、ControlNet、hooks、追加guidance等は未対応です。

API形式の例も同梱しています。

- [プロンプト選択型API](workflows/moss_slider_train_prompts_api.json)
- [手動接続型API](workflows/moss_slider_train_api.json)

**Trainはqueue実行ごとに学習します。** 保存済みLoRAの画像比較には、学習ノードを含めない推論用ワークフローを使用してください。

### 3. 保存したLoRAを適用する

1. `output/moss_sliders/`の`.safetensors`と同名の`.json`を、登録済みのLoRAフォルダー（標準では`ComfyUI/models/loras/`）へ一緒にコピーします。
2. ComfyUIのモデル一覧を更新します。
3. 標準の**Load Diffusion Model**で、学習時と同じチェックポイントを読み込みます。
4. MODELを**MOSS Slider Apply INT8**へ接続し、`lora_name`と`strength`を選びます。
5. ApplyのMODEL出力を通常のMOSS推論ワークフローへ接続します。

```text
Load Diffusion Model → MOSS Slider Apply INT8 → sampler
```

複数のスライダーはApplyノードを連結して使えます。それぞれに学習時と同じ基底のJSONが必要です。

```text
Load Diffusion Model → Apply A → Apply B → sampler
```

強度0はそのノードの追加効果を無効にします。負の強度も使えますが、単方向学習のため、望む逆方向の効果は保証しません。同じseed・promptで強度0 / +1 / +2などを比較してください。

専用Applyは基底のSHA256を照合し、基底違い・JSON欠落・通常のweight patch・bypass以外のinjectionを持つMODELを拒否します。標準LoRA loaderでも論理キーは読めますが、基底一致の検証やINT8重みを保持する適用経路は専用Applyと異なります。

## 学習設定

| 設定 | 既定値 | 内容 |
|---|---|---|
| `steps` | 600 | optimizer更新総数 |
| `rank` / `alpha` | 16 / 2 | LoRA rankと`alpha/rank`係数 |
| `learning_rate` | 0.0002 | AdamWの学習率。weight decayは0 |
| `eta` | 1.5 | teacherの`positive-negative`方向の強さ。samplerのCFGとは別 |
| `seed` | 0 | 学習時の乱数seed |
| `width` / `height` | 768 / 768 | 学習解像度 |
| `rollout_steps` | 16 | native shift=5のEuler scheduleによるnoise rollout |
| `gradient_checkpointing` | true | block単位の再計算で学習時のメモリ使用量を抑制 |
| `blocks_to_swap` | 0 | 末尾N blockのINT8重み・scaleをCPUに保持し、Linear計算ごとに転送。0〜42 |
| `teacher_normalization` | `neutral_l2` | teacher全体のL2ノルムをneutral予測へ合わせる。`none`で無効 |

これらはノードと同梱ワークフローの初期値であり、全モデル・全プロンプトに対する推奨最適値ではありません。`rollout_steps=1`では最大ノイズ時刻だけの学習になります。

### 学習の仕組み

LoRAを無効にした同じ基底から、次のteacherを作ります。

```text
teacher = base(target) + eta × (base(positive) − base(negative))
```

既定の`neutral_l2`では、FP32・sample単位で`teacher *= ||base(neutral)|| / ||teacher||`を適用します。studentはtarget条件・LoRA強度+1で、raw flow velocityのMSEを学習します。

`neutral`がtargetと同じ場合は予測を再利用し、異なる場合はteacherのforwardが1回増えます。L2ノルムが`1e-8`以下なら正規化をスキップし、非有限値では停止します。倍率・参照元・skipの有無はJSONに記録します。neutral正規化は、服装や構図を保持する追加lossではありません。

**以前の未正規化の学習動作を再現する場合は`teacher_normalization=none`を明示してください。** この項目のない既存ワークフロー・APIにも既定の`neutral_l2`が適用されます。保存済みLoRAへの影響はありません。

学習対象は42 block × 4投影の168 adapterです。QKVはAを共有し、保存時にq/k/vへ展開して252論理target・756 tensorのLoRAを出力します。text encoder・text refiner・norm・modulation・入出力投影は学習しません。領域mask、bidirectional学習、画像dataset、学習resumeは未対応です。

## 出力と進捗

保存先は、ComfyUIに設定されたoutputディレクトリの`moss_sliders/`です。フルパスはノードの表示・ログ・reportの`lora_path`と`manifest_path`に表示します。`lora_name`出力はoutputからの相対パスであり、LoRAフォルダーへの自動登録は行いません。

`.safetensors`と同名の`.json`は常に一緒に保持してください。JSONには基底とLoRAのSHA256、設定、実行版、loss、step時間、プロンプトセットの本文とSHA256などを記録します。同名ファイルは上書きせずsuffixを付け、JSONを配置してから完成したLoRAを最終名へ移します。

ComfyUIのコンソール・ログにはエンコード、hash計算、ロード、各step、保存完了が表示されます。

```text
MOSS Slider step 2/6 | 1.30s/step | elapsed 4.6s | ETA 9.2s | loss=0.0239832 grad=0.066817
```

進捗バーはoptimizerの更新数で進みます。ETAは学習ループの目安で、事前のエンコード・hash計算・モデルロードや最後の保存時間は含みません。

## 検証範囲

AMD Radeon AI PRO R9700と`moodyMossMixFreeDownload_v1025STEPSCFG3_int8.safetensors`で、rank 2・1 / 10 stepの学習、保存、強度−1 / 0 / +1での推論を確認しています。小型native MOSSのCPUテストでは、量子化計算・勾配・checkpointing・LoRA mapping・bypass・保存処理・入力検証を扱います。

これは最低必要VRAMや生成品質を示すものではありません。長時間学習や他GPUでの動作、各プロンプトセットの効果は別途確認が必要です。過去の検証環境・条件は[モデル互換性の記録](docs/model-compatibility-20261019.md)を参照してください。記録内のローカルパスや生成物は検証時のもので、配布物には含まれません。

### 開発用のCPU検証

ComfyUIと開発用の`pytest`・`ruff`・`mypy`を用意し、リポジトリ直下で実行します。PythonはComfyUIの依存を利用できる環境を使い、パスは手元の配置に置き換えてください。

```sh
python tools/inspect_checkpoint.py --checkpoint "/path/to/model.safetensors" --metadata-only
python tools/run_cpu_validation.py --comfyui-root "/path/to/ComfyUI" --report "test-results/cpu-validation.json"
```

検査CLIは基底行列を読みません。全ファイルのhashが必要な場合は`--full-hash`を指定します。CPU検証のreportとstageログはローカルの`test-results/`へ出力します。実ComfyUIの依存不足は`BLOCKED`として記録し、GPUの実モデル検証とは区別します。

## 問題が起きた場合

- **ノードが表示されない:** ComfyUIの起動ログでimportエラーを確認し、V3 API・MOSS対応とPython環境の依存関係を確認してください。
- **モデルが一覧にない:** `diffusion_models` / `text_encoders` / `loras`の登録先とファイル名を確認し、一覧を更新してください。
- **Applyでエラーになる:** 学習時と同じ基底を使い、LoRAと同名のJSONが同じフォルダーにあることを確認してください。
- **メモリが不足する:** rank・解像度を下げ、`gradient_checkpointing`や`blocks_to_swap`を調整してください。必要VRAMの下限は未測定です。

不具合報告は[GitHub Issues](https://github.com/Shiba-2-shiba/Comfy-moss-slider-node/issues)へ、ComfyUIの版、OS / GPU、Python / PyTorch、モデル名、再現手順、エラーログを添えてください。

## 参考・第三者ライセンス

MOSS-Image、ComfyUI、Concept Sliders、Krea2 Slider、sd-scriptsなどを参考にしています。参照元と第三者ライセンスは[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)と[licenses/](licenses/)を参照してください。モデル重みの利用条件は、各配布元で確認してください。
