# 実写胸サイズスライダーの調査とプロンプト候補（2026-10-23）

## 目的と変更範囲

胸の変化が出る前に顔や髪が変わるという報告を調べ、実写評価に対応する学習プロンプト候補を追加した。学習処理・Apply処理・既存のアニメ用セット・学習済みLoRAは変更していない。これはプロンプト設計の修正であり、副作用の解消を実画像で確認した結果ではない。

## 確認した証拠

- 学習manifest: ユーザー指定の `D:/StabilityMatrix-win-x64/Data/Models/Lora/PornmasterMoss/breast_size.json`。
- 保存された6レコードはすべて2D animeで、同じ顔立ち・髪型・体格を繰り返す。各ペア内では元から胸サイズだけを変更している。元のペアに無関係な顔の語句差分がある、という不具合ではない。
- 保存されたprompt SHA256と現在の `prompts/breast_size_slider.yaml` のSHA256は一致: `8c8696d3e2e8f70f8b94ae4c12e4c2298f21879106b128aad583a4b0d3159684`。
- 学習設定: 600 steps、rank 16、alpha 2、learning_rate 0.0002、eta 1.5、768×768、rollout_steps 16、seed 500064041625159、teacher_normalization neutral_l2。基底は `pornmasterMoss_v1.safetensors`。
- 全600更新のlossとgradient normは有限。正規化倍率の範囲は0.9743457437〜1.0085104704。数値発散を示す証拠は得られなかった。
- `gazou/moss_image_t2i_00009_.png`〜`00019_.png` は同一成人女性の実写prompt、強度0〜1（0.1刻み）の11枚。別のpromptの `00020_.png`〜`00028_.png` は強度0〜4（0.5刻み）の9枚。各グループ内のprompt graphはLoRA強度を除いて同一。グループをまたぐ比較はpromptが異なる。
- 生成seedは333599710072681、1024×1024、25 steps、CFG 2.5、er_sde/simple、SLG layer 22、start 0/end 0.5。専用 `MossSliderApply` を使用。
- 成人女性の強度0/0.5/1を視認したところ、髪の長さ・目元・輪郭の変化が見え、胸の増大は明瞭でなかった。これは目視所見で、顔類似度や胸サイズの定量測定ではない。

## 原因候補と確度

1. **更新後の優先調査対象**: 胸以外の空間領域に変更を制限する仕組みがない。アニメ調の追加比較でも顔が変わったため、画風の不一致だけでは説明できない。ただし、教師のpositive−negative差分の混入とLoRAの近似誤差の寄与はまだ分離できていない。
2. **補助的な仮説**: 一つの外見への偏り、胸が小さく写る全身構図や輪郭を隠す服・姿勢が、胸方向の学習と汎化を難しくしている。胸の変化を出すため高強度が必要になることと、強度増加に伴う副作用との関係を確認する。画風の不一致は実写で効果が弱い理由の候補には残るが、顔変化の共通原因とは断定できない。
3. **未確定**: 600 stepsでの過学習、rank/learning_rate、全体L2正規化、適用時のsampler/SLGとの相互作用。記録だけでは原因と断定できないため、これらの実装・既定値は変えていない。

`moss_slider_node/training.py` は全latentのvelocity MSEで学習し、`neutral_l2` は教師全体のノルムを調整する。顔を保存する追加lossや領域maskはない。ペアで文章を一致させても、生成時の顔の保持を保証する仕組みではない。

## 追加した候補

`prompts/breast_size_slider_realistic.yaml`: 実写のみ12レコード。6種類の成人女性・髪型・顔立ち・衣装・背景と正面／斜めの構図を使う。不透明のフィットした上着、頭から腰または太もも上までの構図、胴体を遮らない腕の配置とした。各ペアの唯一の変更は `medium-sized breasts` → `large breasts` / `small breasts`。neutralはtargetと同一。

既存アニメ用セットを残し、新セットを明示的に選択する。学習済みLoRAへの自動反映はない。元と異なる出力名で再学習する。600 stepsでは各50更新であり、従来6レコードの各100更新とは異なる。今回の候補は画風・外見の多様性・構図・レコード数を同時に変えるため、画風だけの単一要因実験ではない。

## 次の画像評価

`prompts/breast_size_slider_realistic_eval.txt` の5例を使う。最初に元の成人女性prompt・seed・生成設定で強度0/0.25/0.5/0.75/1を比較し、胸の変化と顔等の変化を別々に記録する。旧LoRAと新LoRAを比較した後、複数seed、未学習衣装、全身、顔アップの対照で確認する。学習設定の追加変更は、この比較結果を得てから一項目ずつ行う。

## 検証と限界

- 新セット未追加時に専用テスト2件が失敗し、追加後に読込・候補一覧・胸サイズ以外のペア本文一致を含む23件が成功した。
- 通常Pythonで単体テスト全件: **206 passed**。`git diff --check` 成功。
- 通常Pythonで全体テスト: 206 passed / 4 failed / 41 errors。ComfyUIの `comfy_aimdo.storage` がその環境に存在せず、統合テストを実行できない。完全ログ: ローカルの `test-results/breast-size-realistic/full-suite-initial.log`。
- ComfyUI venvで通常Pythonのpytestを利用する代替検証も実施したが、21 failed / 189 passed / 41 errors。PyTorchのprecompile artifact重複登録エラーが発生した。完全ログ: `test-results/breast-size-realistic/full-suite-comfy-env.log`。検証環境の修復や依存パッケージ追加は今回の変更に含めていない。
- ruff/mypyは通常Python・ComfyUI venvのいずれにも未導入のため未実行。Pythonテストの構文・import・実行は単体テストで確認した。
- 再学習・新LoRAでの画像生成・顔と胸サイズの定量評価は未実施。改善したとの結論は出していない。

## 追加証拠: アニメ調でも顔が変化（gazou2）

- `gazou2/moss_image_t2i_00023_.png`〜`00031_.png` の9枚を確認した。旧 `PornmasterMoss/breast_size.safetensors` を使い、同じアニメ調prompt・seed 333599710072681・25 steps・CFG 2.5・er_sdeで、強度だけを0/0.5/1/1.5/2/2.5/3/3.5/4に変更している。prompt graphを比較し、強度以外が同一であることを確認した。
- 一覧と強度0/1/4の原画像を視認した。胸の輪郭の増大と同時に、目元・顎や頬の輪郭・髪の形・体格や構図も変化している。目視所見であり、顔同一性や胸サイズの定量測定ではない。
- 学習画風と生成画風が一致する例でも副作用があるため、最初に画風不一致を主な対策対象とした判断を修正する。実写専用セットは実写向けの学習候補として残すが、顔保持の修正が完了したとは扱わない。

### 実装から確認できる範囲

- `moss_slider_node/training.py:81` の `slider_teacher` は基底のtarget予測にpositive−negativeの差分を加える。この差分を胸の空間領域だけに限定する処理はない。本文で胸サイズだけを変えても、モデルの予測差分が胸だけになる保証はない。
- `moss_slider_node/training.py:251` 付近はstudentを強度+1で評価し、全latentのMSEを計算する。+2〜+4は学習した強度より大きく、挙動は未拘束の外挿になる。ただし+1以下の副作用も説明・評価する必要があり、高強度だけの問題とは断定しない。
- `moss_slider_node/lora.py` は全対応blockのqkv/o/ffnへLoRAを注入する。`moss_slider_node/inference.py:119` ではそのLoRAを指定強度で適用し、顔を保護する空間maskや基底画像との合成は行わない。
- neutralはL2ノルムの参照であり、顔を元の予測に固定する損失ではない。

### 次の切り分けと修正方向（未実装）

最初に、同一latent・同一sigmaでteacherのpositive−negative差分と、student−base(target)の差分を比較する。胸と顔の領域を正しく指定して各領域の変化量を測り、教師側の概念の結びつきとLoRAの近似誤差を分けて調べる。sampler/SLGの寄与は設定を一つずつ変更して比較する。

局所化を行う場合の候補は、胸領域の編集maskを各学習構図に合わせ、mask外の教師予測をbase(target)に戻して学習する方法。全体L2正規化を後からそのまま掛けるとmask外の基底予測も変わるため、mask適用と正規化の順序も検証が必要になる。固定の長方形maskを異なる構図へ一律適用して済ませない。これは教師の領域外変更を抑える設計であり、生成画像の顔が完全一致する保証にはならない。

現時点では新たな学習処理変更・再学習・pushは行っていない。設定値だけを変更して直ったと結論づけず、新しい証拠に合わせて調査記録を更新した。
