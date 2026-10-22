# MOSS Slider 学習用プロンプト

このリポの prompts/ に置いた .yaml / .yml を **MOSS Slider Train from Prompts** の prompt_yaml 一覧で選択します。MOSS用の **Load CLIP（type=moss_image）** を接続すると、ファイル内の全レコードをエンコードして学習します。

## 同梱セット

| ファイル | 正のLoRA強度で学習する方向 | 構図 |
|---|---|---|
| [aging_slider_fullbody.yaml](aging_slider_fullbody.yaml) | 成人女性の顔・首・手の皮膚を加齢 | 全身4件、腰上2件 |
| [aging_slider_realistic.yaml](aging_slider_realistic.yaml) | 実写調の成人女性を約35歳から約70歳へ加齢。顔の輪郭・しわ・首・手の変化 | 顔アップ2件、腰上2件、全身2件 |
| [aging_slider_fullbody_male.yaml](aging_slider_fullbody_male.yaml) | 成人男性の顔・首・手の皮膚を加齢 | 全身4件、腰上2件 |
| [deaging_slider_fullbody.yaml](deaging_slider_fullbody.yaml) | 着衣の成人女性から、約4歳の着衣の女児へ。顔と頭身・手足の比率を変える | 全身4件、腰上2件 |
| [deaging_slider_fullbody_male.yaml](deaging_slider_fullbody_male.yaml) | 着衣の成人男性から、約4歳の着衣の男児へ。顔と頭身・手足の比率を変える | 全身4件、腰上2件 |
| [breast_size_slider.yaml](breast_size_slider.yaml) | 着衣の成人女性の胸を大きくする。小さい胸との明示的な対比 | 上半身〜腰下4件、全身2件 |

各ファイル6レコード、合計36レコードです。アニメ調加齢セットでは成人の体格と髪を保ち、皮膚の記述だけを変えています。若返りセットは成人から幼児への年齢・体格変化を学習する内容です。胸サイズではtargetは中程度、positiveは大きい胸、negativeは小さい胸です。アニメ調5セットは2D anime、aging_slider_realistic.yamlは写真の英語キャプションです。同じレコードの衣装・姿勢・背景・光を揃えています。実写加齢では年齢と顔の輪郭の変化も明記し、髪色・髪型は対で保っています。モデル固有のトリガー語は含めていません。

## 書き方と出典

Krea2 Sliderのプロンプトセットにある学習方向と、複数の構図で対を作る方式を参考に、本文をMOSS向けに新規記述しました。参照元は[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)を参照してください。Krea2の学習設定やanchor機能を移植したものではありません。

完成画像に見える人物・外見・服・構図・背景・色・光・画風を自然な英語で記述しています。本文へsystem指示、チャット特殊トークン、品質スコアの羅列を追加しません。MOSS用のテンプレートはnative CLIPが付与します。

## フィールド

~~~yaml
- target: >-
    A 2D anime illustration depicts a fully clothed adult woman
    standing in a softly lit studio, with smooth facial skin.
  positive: >-
    A 2D anime illustration depicts a fully clothed adult woman
    standing in a softly lit studio, with fine facial wrinkles.
  negative: >-
    A 2D anime illustration depicts a fully clothed adult woman
    standing in a softly lit studio, with smooth facial skin.
~~~

- target: studentに入力する基準キャプション。
- positive: 正のLoRA強度で向かう概念。
- negative: teacherの差分を作る比較対象のキャプション。生成用の除外語句を記入する欄ではありません。
- neutral: 教師全体のL2正規化に使用する任意の参照キャプション。省略時はtarget。同梱セットでは全レコードでtargetと同じ本文を明示しています。

通常のYAMLブロック形式・UTF-8 BOM・スカラーのanchor/aliasを読めます。各レコードは必須3フィールドと任意のneutralだけを持つ形式です。neutralを明示する場合は空でない文字列が必要です。anchorによる追加の保持lossは未対応です。未知キー・重複キー・空文字・同一のpositive/negative・フォルダー外のパスは拒否します。読み込みにはComfyUI環境のPyYAMLを使用します。

ノードの`teacher_normalization`は既定`neutral_l2`です。教師全体のL2ノルムをneutral予測へ合わせ、neutral省略時はtarget予測を使います。例として、上記レコードに`neutral: >-`でtargetと同じ本文を追加できます。`none`は従来の未正規化教師で、neutralを読み込んでも追加の教師計算をしません。旧ワークフローやAPIでこの設定を省略した場合もneutral_l2が使われるため、従来の数値動作を比較するときはnoneを明示してください。

教師または参照のL2ノルムが`1e-8`以下なら補正をスキップし、倍率と参照元をレポートに残します。異なるneutralではeta=0でも教師全体を再スケールします。neutral正規化は服装等を保持する追加lossではありません。

ファイルは1〜64レコード、2 MiBまで。各roleの実エンコード結果はbatch1・4096次元・1〜4096tokenのMOSS条件として検査し、paddingを個別に除去します。

## stepsと選択ファイル

stepsはファイル全体でのoptimizer更新総数です。6レコードの場合、0→1→2→3→4→5→0…と一更新につき一件を使います。steps=6で各一回、steps=100なら各16〜17回です。6未満のstepsでは後半のレコードを使いません。

学習のたびにファイルを読み取り、その時点の本文とSHA256をreport/manifestのprompt_sourceへ保存します。同名ファイルを後で編集しても、どの内容で学習したかを追えます。追加ファイルはこのフォルダーに保存し、ComfyUI側のノード定義を再読込すると一覧に現れます。新ノードはqueueごとに学習します。

## 使うワークフローと確認範囲

[編集用workflow](../workflows/moss_slider_train_prompts.json) を開き、checkpoint_name、MOSS用CLIP、prompt_yamlを選択します。[API版](../workflows/moss_slider_train_prompts_api.json) も同梱しています。初期選択は女性若返りの `deaging_slider_fullbody.yaml`、出力名は `deaging_slider` です。学習初期値は **steps=600、rank=16、alpha=2、learning_rate=0.0002、eta=1.5、768×768、rollout_steps=16**、seed=0、gradient_checkpointing=true、blocks_to_swap=0、teacher_normalization=neutral_l2です。ノードを新規追加した場合の既定値も同じです。

ワークフローの基底モデル名は `moodyMossMixFreeDownload_v1025STEPSCFG3_int8.safetensors` です。手元の登録名へ変更してください。600 stepsでは6レコードを各100回使用します。これらは初期値であり、すべてのモデルや学習方向に対する最適値ではありません。強度を上げると、目的の変化に加えて体格・構図・装飾品も変わることがあるため、同じseed・promptで比較してください。

`rollout_steps=1` は最大ノイズ時刻だけの学習になるため避けます。rollout_stepsを増やすと途中状態の生成が入り、1更新あたりの時間も増えます。INT8基底での比較には専用の **MOSS Slider Apply INT8** を使用し、同じseed・promptで強度0/1/2を確認します。実写セットの加齢効果は実画像で未検証です。

手動conditioning型の[編集用workflow](../workflows/moss_slider_train.json)と[API版](../workflows/moss_slider_train_api.json)も同じ学習値・基底に揃えています。こちらは実写加齢の顔アップ1件を初期キャプションにした例です。YAMLの全6レコードを使う場合は、プロンプト選択型を使用してください。保存済みのワークフローはリポジトリの更新では自動変更されないため、同梱版を開き直すか各値を変更してください。

2026-10-19、R9700上のComfyUIで女性加齢の全6レコードを実encoderから6 step学習し、有限loss/gradientとLoRA保存、prompt_sourceのSHA256一致を確認しました。当時の5ファイルの読込・構造はテスト済みですが、各sliderの画質・方向効果・性別分離・構図保持は未評価です。負の強度で望む逆方向の効果が得られる保証もありません。

詳細は[実機検証記録](../docs/prompt-ui-validation-20261019.md)を参照してください。記録内のローカルパスや生成物は検証時のもので、配布物には含まれません。
