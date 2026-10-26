# MOSS Slider 学習用プロンプト

このリポの prompts/ に置いた .yaml / .yml を **MOSS Slider Train from Prompts** の prompt_yaml 一覧で選択します。MOSS用の **Load CLIP（type=moss_image）** を接続すると、ファイル内の全レコードをエンコードして学習します。

## 同梱セット

| ファイル | 正のLoRA強度で学習する方向 | 構図 |
|---|---|---|
| [aging_slider_fullbody.yaml](aging_slider_fullbody.yaml) | 成人女性の顔・首・手の皮膚を加齢 | 全身4件、腰上2件 |
| [aging_slider_realistic.yaml](aging_slider_realistic.yaml) | 実写調の成人女性を約35歳から約70歳へ加齢。顔の輪郭・しわ・首・手の変化 | 顔アップ2件、腰上2件、全身2件 |
| [aging_slider_fullbody_male.yaml](aging_slider_fullbody_male.yaml) | 成人男性の顔・首・手の皮膚を加齢 | 全身4件、腰上2件 |
| [aging_slider_realistic_male.yaml](aging_slider_realistic_male.yaml) | 実写調・成人男性の顔・首・手の皮膚を加齢 | 全身4件、腰上2件 |
| [deaging_slider_fullbody.yaml](deaging_slider_fullbody.yaml) | 着衣の成人女性から、約4歳の着衣の女児へ。顔と頭身・手足の比率を変える | 全身4件、腰上2件 |
| [deaging_slider_fullbody_male.yaml](deaging_slider_fullbody_male.yaml) | 着衣の成人男性から、約4歳の着衣の男児へ。顔と頭身・手足の比率を変える | 全身4件、腰上2件 |
| [deaging_slider_realistic.yaml](deaging_slider_realistic.yaml) | 実写調・着衣の成人女性1人から、約4歳の着衣の女児へ | 全身4件、腰上2件 |
| [deaging_slider_realistic_male.yaml](deaging_slider_realistic_male.yaml) | 実写調・着衣の成人男性1人から、約4歳の着衣の男児へ | 全身4件、腰上2件 |
| [breast_size_slider.yaml](breast_size_slider.yaml) | アニメ調・着衣の成人女性の胸を大きくする。小さい胸との明示的な対比 | 上半身〜腰下4件、全身2件 |
| [breast_size_slider_realistic.yaml](breast_size_slider_realistic.yaml) | 実写調・着衣の成人女性の胸サイズ変化。胸サイズの語句だけを対内で変更 | 6種類の女性・衣装 × 正面／斜めの12件 |
| [hatsune_miku_identity_no_costume.yaml](hatsune_miku_identity_no_costume.yaml) | 初音ミク風の髪・瞳・顔立ちを直接記述。本文は全roleで肯定形 | 6種類の普段着、腰上4件・太もも上まで2件 |
| [hatsune_miku_identity_no_costume_realistic.yaml](hatsune_miku_identity_no_costume_realistic.yaml) | 実写の成人女性を初音ミク風の外見へ。衣装・姿勢・構図は各ペアで同一 | 6種類の普段着、腰上4件・太もも上まで2件 |
| [realistic_female_only_deaging_slider_v2.yaml](realistic_female_only_deaging_slider_v2.yaml) | 実写調・男女2人のうち女性だけを成人から幼児へ変えることを狙う評価用セット | 全身12件、左右と男女の配色を交差 |
| [realistic_female_only_aging_slider_v2.yaml](realistic_female_only_aging_slider_v2.yaml) | 実写調・男女2人のうち女性だけを30代から約70歳へ加齢 | 全身12件、女性限定若返りv2の配色・配置を継承 |
| [realistic_male_only_deaging_slider_v2.yaml](realistic_male_only_deaging_slider_v2.yaml) | 実写調・男女2人のうち男性だけを30代から幼児へ若返り | 全身12件、女性限定若返りv2の配色・配置を継承 |
| [realistic_male_only_aging_slider_v2.yaml](realistic_male_only_aging_slider_v2.yaml) | 実写調・男女2人のうち男性だけを30代から約70歳へ加齢 | 全身12件、女性限定若返りv2の配色・配置を継承 |
| [realistic_clothing_fit_tighten.yaml](realistic_clothing_fit_tighten.yaml) | 実写調・成人女性の服を体に沿うフィットへ。通常fitが基準、ゆったりfitと対比 | 頭から腰下5件、頭から膝下1件 |
| [realistic_clothing_fit_tighten_v2.yaml](realistic_clothing_fit_tighten_v2.yaml) | 実写調・成人女性の服を明示的な密着fitへ。上着・ワンピースに加えてスカート・パンツ、上下両方を収録 | 頭から腰下5件、頭から膝下1件、全身6件 |
| [realistic_clothing_loosen.yaml](realistic_clothing_loosen.yaml) | 実写調・成人女性の服をゆったりしたフィットへ。通常fitが基準、体に沿うfitと対比 | 頭から腰下5件、頭から膝下1件 |

全19セット・合計150レコードです。初音ミク外見と既存の実写衣服フィットを含む13セットは各6レコード、性別限定v2の4セット、実写胸サイズ、実写衣服tighten v2は各12レコードです。アニメ調6セットと実写調13セットから選べます。加齢セットでは成人の体格と髪を保ち、女性の既存実写加齢セットでは年齢と顔の輪郭の変化も明記しています。若返りセットは成人から幼児への年齢・体格変化を学習する内容です。胸サイズではtargetは中程度、positiveは大きい胸、negativeは小さい胸です。同じレコードの衣装・姿勢・背景・光を揃えています。初音ミク外見の修正版はキャラクター名を本文に含めず、外見を直接記述しています。

## MoodyMossなどの実写寄りモデルで使うセット

アニメ調のみだった男性加齢・女性1人の若返り・男性1人の若返り・初音ミク外見に、独立して選択できる実写調YAMLを追加しました。男性加齢と若返りの2セットは、レコード数・順序・学習方向・衣装・姿勢・構図を元のアニメ版から引き継ぎ、写真の英語キャプションへ変更しています。初音ミクはアニメ版・実写版とも6レコードへ整理しました。若返りの実写版は女性だけを変える男女2人の`realistic_female_only_deaging_slider_v2.yaml`とは別で、1人の人物全体を若返らせるセットです。

初音ミク実写版は、比較側・positive側とも着衣の成人女性として記述します。positiveだけに、腰までの青緑のツインテール2本、前髪、青緑の瞳、先細りの輪郭と小さな鼻・口を与えます。キャラ名・作品名・装備の列挙を本文から外し、全roleを肯定形にしています。自然な肌・髪・写真の光と、6種類の普段着・姿勢・構図はペア内で共通です。

1. ComfyUIを再起動し、**MOSS Slider Train from Prompts**の`prompt_yaml`で目的の実写調YAMLを選びます。`checkpoint_name`は学習元にするMoodyMossのファイルを選び、MOSS用CLIPを接続します。
2. `output_prefix`は`moody_deaging_realistic_trial01`や`moody_miku_realistic_trial02`などの別名にします。アニメ版との比較では同じ基底モデル・学習設定を使います。600 stepsなら今回の4セットは各レコード100更新です。更新数の例であり、品質を確認した最適値ではありません。
3. 学習時と同じ基底モデルと専用Applyを使い、同じprompt・seedで強度`0 / 0.5 / 1`を比較します。加齢は皮膚、若返りは顔・頭身・手足、初音ミクは髪・瞳・顔立ちを見て、衣装・構図・画風への副作用も別に記録します。

追加した4セットのLoRA学習評価は未実施です。初音ミク実写版は旧18レコードのpositive生成画像を調べて修正しましたが、修正版の生成は未確認です。実写の文章指定は写真らしさや衣装保持を保証する仕組みではありません。既存LoRAは更新されないため、選択した基底で新たに学習してください。

## 書き方と出典

従来の6セットはKrea2 Sliderの学習方向と、複数の構図で対を作る方式を参考に、本文をMOSS向けに新規記述しました。初音ミク外見セットも本文を新規記述しています。実写女性限定若返りv2は、モデル差を評価する最初の比較用として、元のKrea2用YAMLの全12レコードの本文と順序を維持しています。MOSS向けにヘッダーの設定説明を変更しました。参照元は[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)を参照してください。Krea2の学習設定やanchor機能を移植したものではありません。

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

[編集用workflow](../workflows/moss_slider_train_prompts.json) を開き、checkpoint_name、MOSS用CLIP、prompt_yamlを選択します。[API版](../workflows/moss_slider_train_prompts_api.json) も同梱しています。初期選択は女性若返りの `deaging_slider_fullbody.yaml`、出力名は `moss_slider` です。学習初期値は **steps=600、rank=16、alpha=2、learning_rate=0.0002、eta=1.5、768×768、rollout_steps=16**、seed=0、gradient_checkpointing=true、blocks_to_swap=0、teacher_normalization=neutral_l2です。ノードを新規追加した場合の既定値も同じです。

ワークフローの基底モデル名は `moodyMossMixFreeDownload_v1025STEPSCFG3_int8.safetensors` です。手元の登録名へ変更してください。600 stepsでは6レコードを各100回使用します。これらは初期値であり、すべてのモデルや学習方向に対する最適値ではありません。強度を上げると、目的の変化に加えて体格・構図・装飾品も変わることがあるため、同じseed・promptで比較してください。

`rollout_steps=1` は最大ノイズ時刻だけの学習になるため避けます。rollout_stepsを増やすと途中状態の生成が入り、1更新あたりの時間も増えます。INT8基底での比較には専用の **MOSS Slider Apply INT8** を使用し、同じseed・promptで強度0/1/2を確認します。実写セットの加齢効果は実画像で未検証です。

手動conditioning型の[編集用workflow](../workflows/moss_slider_train.json)と[API版](../workflows/moss_slider_train_api.json)も同じ学習値・基底に揃えています。こちらは実写加齢の顔アップ1件を初期キャプションにした例です。YAMLの全6レコードを使う場合は、プロンプト選択型を使用してください。保存済みのワークフローはリポジトリの更新では自動変更されないため、同梱版を開き直すか各値を変更してください。

2026-10-19、R9700上のComfyUIで女性加齢の全6レコードを実encoderから6 step学習し、有限loss/gradientとLoRA保存、prompt_sourceのSHA256一致を確認しました。当時の5ファイルの読込・構造はテスト済みですが、各sliderの画質・方向効果・性別分離・構図保持は未評価です。負の強度で望む逆方向の効果が得られる保証もありません。

詳細は[実機検証記録](../docs/prompt-ui-validation-20261019.md)を参照してください。記録内のローカルパスや生成物は検証時のもので、配布物には含まれません。

## 初音ミク外見セット（衣装を差分から除外）

`hatsune_miku_identity_no_costume.yaml` と `hatsune_miku_identity_no_costume_realistic.yaml` は、成人女性を初音ミク風の髪・瞳・顔立ちへ近づける6レコードの評価用セットです。前者はアニメ、後者は実写のキャプションです。本文にはキャラ名・作品名を入れず、positiveに青緑の瞳、前髪、腰までの高いツインテール2本、先細りの輪郭と小さな鼻・口を記述します。target・negative・neutralには同一の比較人物を使います。YAMLのスカラーanchor/aliasは本文の共有であり、追加の学習機能ではありません。

クリーム色のセーター、赤茶色のブラウス、紺色のパーカー、オリーブ色のカーディガン、ラベンダー色のワンピース、デニムジャケットを各1レコード使います。腰上4件・太もも上まで2件で顔と髪の占有面積を確保し、近接構図と靴・足元までの描写を同時に要求しない構成です。比較側の髪型・目・輪郭は3種類を各2件に配分しています。衣装・姿勢・表情・構図・背景・光は各ペアの全roleで同一です。本文はすべて肯定形で、`negative`も生成用の除外語句ではなく比較人物の肯定形キャプションです。

旧実写版のpositive画像では、1番でツインテールの過剰な増殖と全身化、別の画像でマイク・手袋の混入が見られました。極端な長髪と近接構図の競合、キャラ名からの装備の連想が候補です。修正版はその競合を減らすための内容変更であり、衣装保持の追加loss・領域maskではありません。今回の画像はLoRA適用前の生成なので、学習不足の証拠には使えません。調査と限界は[初音ミク修正記録](../docs/miku-prompt-refinement-20261024.md)を参照してください。

1. ComfyUIを再起動し、**MOSS Slider Train from Prompts** の `prompt_yaml` で目的のアニメ版／実写版を選びます。出力名は `moss_hatsune_miku_identity_trial02` など既存と異なる名前にします。
2. `teacher_normalization=neutral_l2` を使う場合、neutralはtargetと同じ予測を再利用します。stepsは全体の更新総数です。旧18件では600 stepsで各33〜34更新、修正後の6件では各100更新、300 stepsでは各50更新です。各更新が共通の外見方向を学ぶため、総更新数が同じなら18件版の外見学習回数そのものが3分の1になるわけではありません。件数削減は構図と条件の分散を減らす選択で、効果が3倍になる保証ではありません。初回はsteps以外の設定を元の試行に揃え、効果を確認してから1項目ずつ調整します。
3. 学習時と同じ基底と **MOSS Slider Apply INT8** を使い、同じprompt・seedで強度 `0 / 0.5 / 1` を比較します。まずtargetの本文を生成に使い、キャラ名を指定しなくても外見の方向が現れるかを見ます。髪・瞳・顔立ちの変化と、衣装・構図の変化を別々に評価します。
4. まず各レコードのtarget／positiveをLoRA強度0で生成し、顔と髪の描写・髪束の本数・構図・衣装が意図に沿うかを複数seedで確認します。その後、学習済みLoRAを未学習の衣装でも比較します。下記はアニメ版の例で、実写版では冒頭を`A realistic color photograph`へ変更します。修正版での再生成・学習効果は未検証です。

```text
A 2D anime illustration depicts one fully clothed adult woman with shoulder-length brown
hair and brown eyes. She wears a plain burgundy coat over a white blouse,
beige trousers, and brown ankle boots. A full-body three-quarter view shows
her standing at a quiet train station with a neutral expression.
Soft daylight illuminates her face, hair, and plain everyday clothing.
```

## 女性限定若返りv2の評価

`realistic_female_only_deaging_slider_v2.yaml` はMOSS-Imageでの採否を決めるための初期候補です。全12レコードで `target == negative == neutral` とし、`positive` では女性の年齢・顔・頭身・手足だけを幼児へ変えています。男性の記述は各ペアで同一です。3組の配色について、男女への色の割当と左右配置を入れ替えています。男性の保持や性別分離を保証するマスク・追加lossはありません。ユーザーから期待した効果が得られたとの報告があります。今回、その画像・学習設定の独立した再検証は実施していません。

1. ComfyUIを再起動して、Trainノードの `prompt_yaml` にこのファイルを選択します。`output_prefix` は `moss_realistic_female_only_deaging_v2_trial01` などの別名にし、既存LoRAと区別します。
2. 初回は上記のMOSSノード既定値で学習します。Krea2の `single`、`teacher_guidance_scale`、`teacher_norm_reference` はこのノードの設定ではありません。600 stepsでは各レコード50更新です。既存6レコードセットと同じ総更新数でも、1レコードあたりの更新数は半分になります。
3. 同じ基底モデルと **MOSS Slider Apply INT8** を使い、成人男女を描く同一prompt・seedで強度 `0 / 0.5 / 1 / 2` を比較します。各強度で同じ初期ノイズを使い、生成用seedの制御は `fixed` にします。複数seedで繰り返し、生成設定と学習manifestを保存します。効果を見るため、比較用promptの女性は成人のまま記述します。
4. 学習レコードのtargetに加え、未学習の服色・背景、左右を入れ替えた構図、女性1人、男性1人でも確認します。女性の顔と全身の幼児化、男性の年齢・身長・体格、服色・着衣状態、人数・左右・全身の収まり、写真らしさを分けて記録します。

女性の変化が複数seedで再現し、男性と構図の変化が許容範囲なら採用候補とします。男性まで若返る場合は性別分離、色や左右で効き方が変わる場合は配色・配置への依存を優先して見直します。顔だけ変わって体格が残る場合は全身比率の対比を調整します。強度2だけで崩れる場合は低強度での実用性を先に判断します。次の試行では本文か学習設定の一方だけを変え、別名で保存します。

## 性別限定v2の追加3セット

ユーザーから期待した効果が得られたと報告された `realistic_female_only_deaging_slider_v2.yaml` を基に、女性のみ加齢・男性のみ若返り・男性のみ加齢を追加しました。元セットの12レコードの順序、3組の配色と男女への色の割当、左右配置、衣装、髪型、全身構図、背景、光を引き継いでいます。元の女性限定若返りYAMLは変更していません。

- `realistic_female_only_aging_slider_v2.yaml`: 女性は30代の滑らかな肌から約70歳へ。顔のしわ、頬・顎の輪郭、首・手の皮膚を変え、成人の身長・頭身・手足と髪型・髪色を保ちます。男性は元セットの30代の記述を維持します。
- `realistic_male_only_deaging_slider_v2.yaml`: 男性は30代から幼児へ。幼い顔、大きな頭の比率、低い身長、小さな肩・手、短い手足を記述します。女性は元セットの成人の記述を維持します。服・靴は各自の体格に合うサイズとして対内で共通です。
- `realistic_male_only_aging_slider_v2.yaml`: 男性は30代から約70歳へ。顔のしわ、頬・顎の輪郭、首・手の皮膚を変え、成人の身長・頭身・手足と短い黒髪・ひげのない顔を保ちます。女性は元セットの成人の記述を維持します。

全件で `target == negative == neutral` です。positiveでは対象の人物の年齢・外見だけを変更し、相手の人物・衣装・姿勢・背景などの本文は一致させています。性別限定は学習意図であり、マスクや保持lossを追加するものではありません。

ComfyUIを再起動またはノード定義を再読込し、`prompt_yaml`で追加ファイルを選択してください。初回は効果が得られた元セットの基底モデル・学習設定を使い、`output_prefix`だけセットごとの新しい名前にします。各12件なので600 stepsなら各50更新です。既存LoRAは更新されないため、各セットで新たに学習します。

同一prompt・seed・生成設定で強度 `0 / 0.5 / 1` を比較し、対象人物の変化と相手の年齢・体格の保持を分けて確認します。左右を入れ替えた構図や未学習の服色でも確認してください。追加3セットの読込・本文整合性と、学習後の効果は別の検証です。追加版の学習・生成画像評価は未実施です。

## 実写衣服フィットの2セット

Krea2用の `clothing_fit_tighten_slider_v2.yaml` と `clothing_fit_loosen_slider_v2.yaml` を基に、全roleのアニメ・セル塗り表現を、自然な肌と布地を描く実写のスタジオ写真へ変更しました。MOSS用native CLIPを接続し、Trainノードの `prompt_yaml` で選択します。Krea2の `training_direction=single` はMOSSノードの設定ではありません。

| ファイル | target / neutral | positive（正のLoRA強度の方向） | negative（教師の比較対象） |
|---|---|---|---|
| `realistic_clothing_fit_tighten.yaml` | 通常のフィット | 体に沿うフィット | ゆとりの大きいフィット |
| `realistic_clothing_loosen.yaml` | 通常のフィット | ゆとりの大きいフィット | 体に沿うフィット |

各6件で、ブラウス、カーディガン、ジャケット、セーター、トップス、ワンピースを元の順序で使います。2ファイルのtargetは同一で、positiveとnegativeを入れ替えています。対内で異なるのは服と体の隙間・腰の輪郭・袖幅を記述する2文だけです。成人女性の体格、髪、衣装の種類・色・着衣状態、姿勢、構図、背景、光は共通です。カーディガンとジャケットはインナーの上からのフィットを記述します。

腕を少し体から離し、上着は頭から腰下まで、ワンピースは頭から膝下までを写して、服の輪郭と裾を見せます。対象は衣服のゆとりであり、胸サイズや体型そのものの変更ではありません。neutralはtargetと同じ通常fitのL2正規化参照で、体格・顔を固定する追加lossやマスクではありません。

初回は比較したいMOSS基底モデルと学習設定を揃え、各セットで別の `output_prefix` を使って学習します。600 stepsでは各レコード100更新です。同じprompt・seed・生成設定で強度 `0 / 0.5 / 1` を比較し、布と胴体の隙間、腰の外形、袖幅を確認します。顔・髪・体格・衣装の種類と色への副作用も確認してください。新2セットの学習・生成画像評価は未実施です。

## 実写衣服tighten v2（上着・スカート・パンツ）

`realistic_clothing_fit_tighten_v2.yaml`は、旧tightenで強度4〜5まで効果が見えにくいという報告を受けた改善候補です。旧版は残し、positiveに`tight-fitting`、`body-hugging`、`snug`、`taut fabric`と、細い袖・服の外形が体に沿う描写を追加しています。低強度での効果を確認済みという意味ではありません。

全12レコードの内訳は、ブラウス・カーディガン・ジャケット・セーター・トップス・ワンピースの6件、スカート2件、パンツ（トラウザーズ・ジーンズ）2件、ブラウス＋スカートとセーター＋パンツの2件です。下半身と上下両方の6件は全身構図で、腰・ヒップ・太もも・裾、パンツではふくらはぎと左右の脚の輪郭も見せます。スカートは一続きの外形を記述します。

各レコードのtarget／neutralは通常fit、positiveは密着fit、negativeはゆったりfitです。変えるのはフィットの2文だけで、35歳の成人女性、頭身・肩幅・胴・手足、服の種類・素材・色・丈、姿勢・構図・光は共通です。上着だけの例では下衣、下衣だけの例では上着の記述を揃え、上下両方の例では両方のフィットを変えます。これは保持の学習意図であり、顔・体格を固定するlossや領域mask、生成時に対象衣服を選ぶ仕組みではありません。未学習の衣服への適用や体型保持は画像で確認する必要があります。

ComfyUIのノード定義を再読込または再起動し、Trainの`prompt_yaml`で新ファイルを選んで、`output_prefix`を`moss_clothing_fit_tighten_v2_trial01`などの別名にして再学習します。既存LoRAはYAML追加だけでは変わりません。12件なので600 stepsでは各50更新、1200 stepsでは旧6件・600 stepsと同じ各100更新です。最適値を確認したものではないため、総更新数を揃える比較と、1件あたりの更新数を揃える比較を区別してください。

同じ基底モデル・生成prompt・seed・設定で、強度`0 / 0.5 / 1 / 2`を比較します。生成promptは通常fitの成人女性を記述し、上着、スカート、パンツ、上下両方を別々に確認します。密着の変化に加え、顔・体格・服の種類・素材・丈・着衣状態への副作用を確認します。追加版の読込と本文整合性の確認は、学習後の低強度での効果検証とは別です。

## 実写胸サイズの評価

`breast_size_slider_realistic.yaml`は、アニメ調の胸サイズセットで学習したLoRAを実写に適用すると、胸の変化より先に顔や髪が変わるという報告を受けて追加した実写専用の候補です。既存の`breast_size_slider.yaml`はアニメ調の用途に残しています。

新セットは6種類の成人女性・髪型・顔立ち・衣装・背景を、それぞれ正面と斜めの構図で記述します。胸の輪郭を観察しやすい不透明のフィットした上着と、頭から腰／太もも上までの構図を使います。各レコードではtargetの`medium-sized breasts`をpositiveで`large breasts`、negativeで`small breasts`に変えるだけで、残りの本文は同一です。neutralもtargetと同じです。

画風の不一致と一つの外見への偏りを減らす設計ですが、顔や服装を固定する追加loss・マスクではありません。追加のアニメ調比較でも旧LoRAによる顔の変化が確認されており、画風の不一致だけでは説明できません。このセットの追加で顔保持を修正できたとは扱っていません。画像での改善は未確認です。学習済みLoRAはYAMLの変更で更新されないため、新たに学習する必要があります。

1. ComfyUIのノード定義を再読込し、Trainの`prompt_yaml`で`breast_size_slider_realistic.yaml`を選びます。`output_prefix`は`breast_size_realistic_trial01`など既存と異なる名前にします。
2. 最初の比較では基底モデルと学習設定を元の試行に揃え、プロンプトセットを変更します。元の報告は600 steps・rank 16・alpha 2・learning_rate 0.0002・eta 1.5・768×768・rollout_steps 16・teacher_normalization neutral_l2です。600 stepsは12件で各50更新となり、従来の6件で各100更新とは異なります。品質を確認した最適値ではありません。
3. 学習時と同じ基底と専用Applyを使い、同一prompt・seed・生成設定で強度`0 / 0.25 / 0.5 / 0.75 / 1`を比較します。[positive prompt例](breast_size_slider_realistic_eval.txt)は成人女性1人のみを記述しています。最初は元画像と同じseed・sampler・CFG・SLGを保ち、その後に複数seedへ広げます。
4. 胸の輪郭の変化と、顔・髪・肩幅・胴回り・衣装・構図の変化を分けて評価します。顔だけを描く対照でも副作用を確認します。低強度から副作用が先行する場合、強度を上げ続けず、学習方法の追加検討に進みます。

元画像・学習manifestの調査範囲と仮説は[調査記録](../docs/breast-size-realistic-20261023.md)を参照してください。
