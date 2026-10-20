# MOSS Slider LoRA 改善方針

作成日: 2026-10-20
状態: 方針案。製品コードの変更・再学習は未実施。

## 目的と判断基準

目的は、同程度の老化を実現したときの衣服・姿勢・背景・人物の特徴への副作用を減らすこと。
LoRA強度の表示値を小さくすることや、学習lossを下げることだけを改善とは扱わない。
現在のテキストによる学習、凍結INT8基底、専用Applyによるbypass、既存LoRAとの互換性を維持しながら段階的に検証する。

今回の作業は方針検討まで。以下の新設定や機能は実装済みではない。

## 現在までの証拠

| 実験 | 主な学習条件 | 同一生成条件での目視評価 |
|---|---|---|
| 旧版・専用Apply | 300 steps、rank16、alpha2、eta1、lr1e-4、rollout8、seed0 | 強度4～5で老化。5では脚の肌を維持、6で脚の衣服変化。姿勢変化はある |
| 追加生成２ | 600 steps、alpha2、rollout8、seed426173177313119 | 4までは老化が弱く、5で老化と黒い脚の衣服が現れる |
| 追加生成３ | 300 steps、alpha4、rollout16、seed451375103277716 | 4から明確な老化と脚の衣服変化。5で老化が強い |

参照:
- `../../../生成Loraと画像/moss_aging_slider.json`
- `../../../追加生成/`
- `../../../追加生成２/moss_aging_sliderV2.json`
- `../../../追加生成３/moss_aging_sliderV2_001.json`

学習プロンプト6組と基底SHA256は共通。生成に到達するノードを辿ると、旧版の「追加生成」と新２・３はLoRA名・強度を除いて同じ条件で、SLGは生成経路外だった。
旧版と新２の強度0画像は画素単位で一致した。新２・３のフォルダーにはLoRA本体がなく、その重みのハッシュは独立検証できていない。
学習seedが全版で異なるため、steps・alpha・rolloutの個別効果は確定していない。

実装で確認したこと:
- 現在は `base(target) + eta * (base(positive) - base(negative))` を教師とする。全6組でtarget=negative、eta=1なので、この条件では教師は数式上positive予測になる。`moss_slider_node/training.py:78`
- 学習は全体のraw velocity MSE、student強度+1のみ。教師正規化・独立した保持損失はない。`moss_slider_node/training.py:172`
- 42ブロックのqkv・o・ffn_up_gate・ffn_downを対象とする。`moss_slider_node/lora.py:13`
- seedはLoRA初期化と時刻・ノイズ列の両方に使用される。`moss_slider_node/job.py:71`、`moss_slider_node/training.py:124`
- 設定・基底hash・プロンプト本文・対象層・実行ライブラリ版は既に記録される。これを再利用する。`moss_slider_node/job.py:89`
- JSONを先に確定し、LoRAの最終名へのrenameを完了点とする。manifestなしの完成LoRAを防ぐが、中断時に孤立JSONは残り得る。専用Applyでhash・基底一致を検査する。`moss_slider_node/export.py:70`、`moss_slider_node/inference.py:60`

参考実装の差:
- Krea2はneutralを受け付けるが、既定のteacher_norm_reference=noneでは正規化に使用しない。neutral指定時はL2ノルムを合わせる。`../../../Confyui-krea2-slider-node/krea2_slider_node/training.py:94`、同`slider_loss.py:5`
- ai-toolkitは教師の平均・標準偏差を基準予測に合わせ、正負両方向と任意anchorを学習する。入力latentもデータセット由来であり、MOSSのテキストrolloutと同一ではない。`../../../ai-toolkit/extensions_built_in/concept_slider/ConceptSliderTrainer.py:199`
- neutralによる教師正規化と、anchorによる基底予測保持は別機能。どちらも衣服保持の保証ではない。

## 選択肢と推奨

1. 設定値だけの探索: 実装負担は小さいが、現在の比較ではseedや条件が混在しており、GPU時間を増やしても原因を識別しにくい。
2. 評価・プロンプト改善を先行し、必要な学習機能を選択式で追加: 推奨。既存資産を使って原因を絞り、効果を確認した変更だけを採用できる。
3. ai-toolkit方式へ全面変更: データ経路、正規化、双方向、anchorが同時に変わる。現段階の原因切り分けには不向きで、別企画とする。

## 段階0: 比較を再現可能にする

最初に、既存3版を使った評価セットを定義する。実行前に各LoRA本体を特定し、レポートのhashと一致することを確認する。
新２・３の重みを入手できない間は既存PNGの評価に限定し、新プロンプトの比較を実行済みとはしない。
利用可能な重みでの再生成には新しい学習は必要としない。

- 学習文そのもの、意味を維持した言い換え、未学習の衣服・姿勢を別々に評価する。
- 年齢と衣服・姿勢が結び付かないよう、同じ人物条件でスカート／ズボン、座位／立位／自撮り等を交差させる。
- 最終比較用の未学習プロンプトは学習YAMLへ混ぜない。
- 生成seed・prompt・negative prompt・モデル・sampler・scheduler・CFG・解像度・専用Apply・強度範囲を固定する。
- まず1seedの少数ケースで候補を絞り、候補のみ3seed×3未学習プロンプトで確認する。
- 強度0/1/2/3/4/5を通常比較の出発点とし、既知の衣服変化を追跡する強度6もストレス確認点として残す。同程度の老化に合わせる必要があるときだけ中間強度を追加する。

追加する比較補助ツールの責務:
- レポートとPNG埋め込み情報から差分表を作り、画像保存ノードに到達する接続だけを比較する。
- 学習seedと生成seed、alphaと推論強度、学習etaと生成CFGを区別して表示する。
- LoRA本体が存在すればhashを確認し、欠けていれば「未検証」と出す。
- 画像、設定、レポート、コード版を実験IDで対応付ける。既存manifestの情報を重複実装しない。

目視評価は老化・衣服・姿勢・背景・人物の特徴を別項目で記録する。年齢推定器や単純な画素差を採否の主指標にはしない。
衣服は素材の微差、色柄・裾の変化、種類の変化を区別する。
暫定採用基準は、同程度の老化に揃えた9ケース中7ケース以上で衣服・姿勢の悪化がなく、過半数で少なくとも一方が改善すること。
新たな衣服種類の変更など重大な副作用が増えた候補は採用しない。この少数試験で一般的な最適性を主張しない。

予定ファイル: `tools/compare_slider_runs.py`、`docs/evaluation/`、`workflows/`。

## 段階0b: 学習診断と途中保存

現在のlossだけでは、老化の教師信号が小さいのか、studentが追従できていないのかを区別できない。
`training.py`で既に求めた予測を再利用し、以下を任意で記録する。

- `teacher - base(target)` と `student - base(target)` のRMS。
- 差分に対する相対MSEとcosine。差分がほぼ0の場合は無効値として明示し、分母を小さくして巨大な数を作らない。
- 正規化導入後は補正前後の差分RMSと補正係数。
- record・実際のsigma別の集計。オンライン値はランダムサンプル上の診断であり、検証lossとは呼ばない。

同じ固定ノイズ・時刻・条件による検証probeを別途用意し、学習に使った条件と未学習条件を分ける。
検証は独立した乱数列を使い、学習の乱数消費・勾配・LoRA multiplier・モデルのモードを変えない。

任意の途中LoRA保存を追加し、100/300/600更新等を同じ学習軌跡で比較可能にする。
各snapshotは既存のJSON先行・LoRA最終renameによる保存手順を使い、実際の完了stepと共通run IDを記録する。
途中保存でもmanifestなしの完成LoRAを公開しないことを検証する。中断時の孤立JSONは未完成として区別する。
最終保存のみを既定とし、optimizerを含むresumeはこの段階に含めない。

seedの内訳は初期化とサンプリングを区別して記録する。将来別seed指定を導入する場合も、未指定時は従来のseedで両方を駆動して互換性を保つ。
同一設定の繰り返し検証なしに、seed差を設定差の効果と呼ばない。

予定ファイル: `moss_slider_node/training.py`、`job.py`、`export.py`、`nodes.py`、既存training/job/exportテスト。

## 段階1: プロンプトのみの対照実験

最初の新規学習は、既存設定を固定したプロンプト改訂の比較を優先する。
現行YAMLを残し、別名のv2として追加する。

- 老若ペア内の衣服・姿勢・画風・背景は同一にする。
- 同じ人物・構図に対して衣服を交差させ、同じ衣服に対して姿勢を交差させる。
- 顔アップと身体を含む構図の割合を明示する。
- 「人物全体を35歳から70歳にする」記述と「成人の体格を保ち、顔・首・手の老化表現を変える」記述を混ぜず、比較仮説を明示する。
- 固定年齢を除くことやモデル固有トリガーを足すこと自体の効果は未確定。必要なら別実験に分ける。
- レコード数を増やす場合は総更新数と各レコードの更新数を両方記録する。データ多様化と反復回数の変化が混在したことを明示する。

同一seedのLoRAなしで老若プロンプトも生成し、教師モデル側で衣服・構図が変わるか確認する。
これはteacherのvelocityを直接評価する検証と区別して扱う。

予定ファイル: `prompts/aging_slider_realistic_v2.yaml`、`prompts/README.md`、独立した評価prompt定義。

## 段階2: 教師正規化と学習対象を別々に検証

### 2A. 任意neutralとL2正規化

- YAMLと手動conditioningに任意neutralを追加し、省略時はtargetを参照する。
- `teacher_normalization=none|neutral_l2`を追加。既定はnone。
- 先に教師全体 `T = base(target) + eta * (positive - negative)` を作り、L2モードで `T * ||neutral|| / ||T||` とする。
- FP32、sampleごとの演算とし、ゼロノルム等の縮退条件を明示的に処理・報告する。教師は常にdetachする。
- neutralキーの追加だけでは正規化を有効にしない。
- これはanchor lossではない。平均・標準偏差を合わせるai-toolkit型は、L2の評価後の独立した実験とする。
- etaと重複するteacher_guidance_scaleは追加しない。方向の係数はetaに一本化する。

noneでは従来の教師・loss・乱数列・更新結果を維持する。
正規化は弱い／不安定な予測を増幅する可能性もあり、既定化は固定probeと画像比較を通過した場合に限る。

予定ファイル: `training.py`、`prompt_files.py`、`job.py`、`nodes.py`、training/prompt/nodeテスト。

### 2B. Attention限定の対象選択

- `target_modules=all|attention`を追加。既定は従来のall。
- attentionは全42ブロックのqkvとoを対象にし、FFNを除く。特定層番号の探索は後回しにする。
- all: 168内部adapter、252論理target、756tensor。
- attention: 84内部adapter、168論理target（q/k/v/o）、504tensor。
- target profileをmanifestとsafetensors metadataへ明示し、実際のキー・shape・QKV共有Aを厳密に照合する。
- 旧artifactのprofile未記載はallとして扱う。FFNが欠けた旧artifactをattentionと推測して受理しない。
- 現在のApplyは全projectionを必須としているため、学習側だけでなく保存・読み込み・bypass適用を一緒に対応する。

根拠となる現行契約: `moss_slider_node/inference.py:19`、`:100`、`tests/integration/test_inference.py:29`。
Attention限定で副作用が減ることは仮説であり、老化能力も落ちる可能性がある。
最初はall/none、all/neutral_l2、attention/noneを独立に比較し、改善が確認された場合だけ組み合わせる。

予定ファイル: `training.py`、`lora.py`、`job.py`、`export.py`、`inference.py`、`nodes.py`、既存LoRA/inferenceテスト。

## 設定探索の扱い

lr・eta・alphaの探索は、評価条件と対象・正規化を固定した枝で行う。
既存の良好な結果を比較基準に残し、各枝で一項目だけを変更する。

- lr: 1e-4を基準に2e-4を比較。必要なら5e-5も比較するが、更新速度と選択性を混同しない。
- eta: 1を基準に2を比較。ai-toolkitの既定3をそのまま移植しない。
- alpha: rank16のまま2と4を比較。学習後alphaの書換えを品質改善と扱わない。
- 300→600は同じseed・同じ軌跡のsnapshotで比較。画像と固定probeの根拠なく1200へ延長しない。
- rollout8/16も独立比較とし、sigmaの範囲と学習時間を記録する。

この一覧は実験候補であり、全組合せを一括で走らせる計画ではない。

## 段階3: 保持損失は必要性が確認されてから

固定probeで教師方向への追従が良く、未学習画像でも老化と衣服変更が結び付く場合に検討する。

- anchorは独立したconditioningと基底教師を持ち、係数0を完全な無効状態とする。
- 空文字を指定しただけで暗黙に有効化しない。
- 同じlatentを使うか別のanchor trajectoryを使うか、時刻・適用頻度・lossのスケールを先に定義する。
- 広い「person」を保持すると老化の学習自体と競合し得る。anchorの対象と評価ケースを対で設計する。
- 双方向学習は正方向の品質が確認された後の任意機能。agingの逆方向を自動で望ましい若返りとみなさない。
- 顔以外の厳密な画像保持が必要なら、maskや画像ペアを使う別段階が必要。テキストonlyのneutral/anchorだけで保証しない。

## 実装・採用の検証条件

1. 旧YAML、旧API入力、旧LoRA/manifestが引き続き使える。
2. none/all・診断OFF・途中保存OFFで、固定CPU toyの教師・勾配・数値更新・乱数列が従来と一致する。
3. 診断・固定probe・途中保存を有効にしても、その有無で学習の数値結果が変わらない（時刻等の実行メタデータは除く）。
4. neutralの省略／明示、同一文の共有、ゼロノルム、非有限値、無効設定、cancelを検証する。
5. 全対象とattention対象の保存・再読込・専用bypassを検証する。欠落／余分なtensor、profile不一致、基底hash不一致は拒否する。
6. 強度0が基底と一致し、±適用後もINT8基底が不変である。数値互換と画像品質を別の合否にする。
7. pytestの対象テストから開始し、変更範囲に応じて全CPU suite、ruff、mypy、実機の短い学習・保存・Apply確認を行う。
8. 新しい既定値の採用は、固定評価セットの画像基準を通過してから行う。学習成功やloss低下だけでは昇格させない。

## 最初の実装単位

評価セット・比較ツール、診断／固定probe、途中保存を小さな変更単位に分ける。
その証拠に基づくプロンプトv2比較を先行し、neutral L2、対象選択を別々の変更として進める。
anchor、双方向、自由な層指定、画像学習、完全resume、optimizer追加は初回実装にまとめない。
新規依存は原則追加せず、既存のComfyUI・PyTorch・safetensors・YAML処理を使用する。
