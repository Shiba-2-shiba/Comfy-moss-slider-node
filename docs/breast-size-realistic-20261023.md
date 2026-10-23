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

1. **有力な仮説**: 学習はアニメ、評価は実写という領域の不一致。不一致自体は確認済みだが、それが顔変化を生むという因果は未確認。実写専用セットでの再学習が次の比較になる。
2. **補助的な仮説**: 一つの外見への偏り、胸が小さく写る全身構図や輪郭を隠す服・姿勢が、実写での胸方向の学習と汎化を難しくしている。外見・衣装・構図を変えた候補で検証する。
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
