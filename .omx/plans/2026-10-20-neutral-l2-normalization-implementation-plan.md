# Optional Neutral L2 Normalization Implementation Plan

> **実装時のユーザー指示による変更（2026-10-20）:** 既定値を`neutral_l2`とし、既存6YAML・全36レコードへtargetと同じneutralを追加する。旧APIの項目省略もneutral_l2となる。従来の数値動作は明示`none`で検証する。以下の旧「既定none」「既存YAML本文・hash不変」要件はこの指示で上書きされる。target/positive/negativeの既存本文は維持する。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Native execution is recommended because the input, training, and node interfaces are tightly coupled.

**Goal:** MOSS Slider学習に任意のneutral条件と、明示選択時だけ有効になるKrea2型の教師L2正規化を追加する。

**Architecture:** 既存の`training.py`に正規化関数と設定を追加し、YAML／手動CONDITIONINGから任意neutralを運ぶ。既定`none`では従来の教師演算と学習経路を維持し、正規化のモード・参照元・係数を既存JSONレポートへ記録する。LoRAの対象層・保存tensor形式・Applyは変更しない。

**Tech Stack:** Python 3.10以上、PyTorch、ComfyUI V3、既存PyYAML、safetensors、pytest。新規依存なし。

**Spec:** [改善方針・段階2A](2026-10-20-aging-slider-quality-roadmap.md)。本書で今回の詳細仕様を確定する。全体方針の他段階は今回の実装に含めない。

**Baseline:** `c19b5ec818b5521731d5b0de2e69c8ff56390ad9`。作成日2026-10-20。本書は計画のみで、製品コードの変更・テスト実行・再学習は未実施。

## Global Constraints

- `teacher_normalization`の選択肢は`none`と`neutral_l2`、既定は`none`。
- 必須条件は従来どおりtarget・positive・negative。neutralは任意で、省略時だけtargetを参照する。
- 旧3フィールドYAML、旧API引数、旧ワークフロー、旧LoRA/manifestを引き続き使用できること。
- FP32 LoRA、凍結INT8基底、student強度+1、現在のloss・optimizer・時刻抽出・乱数列を維持する。
- noneでは従来の`slider_teacher`の演算順・dtypeを変えず、neutralの教師forwardも正規化も実行しない。
- 既存の全6YAMLの本文とSHA256、学習プリセットのsteps/rank/alpha/lr/eta等を変更しない。
- anchor loss、mean/std正規化、Attention限定学習、双方向学習、途中保存、固定probe、resumeは今回の対象外。
- neutral正規化の動作確認と、画像品質・副作用改善の実証は別の完了条件とする。

## Review Focus

以下の5項目を対応タスクのテストに含める。

1. 新しい設定を持たない旧API／旧ワークフローが、入力検証段階で拒否されないこと（Task 3）。
2. noneでneutralを与えても教師計算・乱数列・学習結果に影響しないこと。ただし入力自体の妥当性検査は行う（Task 2）。
3. neutral省略・明示・混在レコードで参照元を誤らず、進捗の分母も合うこと（Task 2、3）。
4. ゼロ／微小ノルム・非有限値・極端な有限値でNaNや暗黙の過大増幅を発生させないこと（Task 1）。
5. 正規化中のcancelやエラー後も学習状態とjob lockが復旧し、旧／新manifestのLoRAをApplyできること（Task 4）。

## 現行実装の境界

| ファイル | 現状と今回の変更責務 |
|---|---|
| `moss_slider_node/training.py:14` | TrainingConfig、教師式、3条件の検査、学習ループ。設定・L2関数・任意参照・記録を追加 |
| `moss_slider_node/prompt_files.py:42` | YAMLを3キーに限定。任意neutralを許可し、encode進捗を実際のrole数にする |
| `moss_slider_node/job.py:47` | conditioningキーを3つに限定。4つ目の任意neutralを通常の検査に通す |
| `nodes.py:74` | 手動学習ノードのschemaとexecuteに末尾追加。旧入力順を維持 |
| `nodes.py:115` | YAMLノードはschemaを継承。手動neutral入力を除外し、同じ正規化設定を使う |
| `moss_slider_node/export.py:76` | config/training情報を既存manifestへ保存できる。基本的に変更不要 |
| `moss_slider_node/inference.py:60` | 学習設定の追加項目には依存しない。基本的に変更不要、回帰検証のみ |
| `tests/unit/test_training.py` | 数値演算・教師detach・乱数再現の既存テストを拡張 |
| `tests/unit/test_prompt_files.py` | YAML検査・encoder cache・既存preset数のテストを拡張 |
| `tests/integration/test_job.py`、`test_nodes.py`、`test_inference.py` | 入力から保存・Applyまでの契約を検証 |

Krea2の参照箇所は`../../../Confyui-krea2-slider-node/krea2_slider_node/slider_loss.py:5`。
通常域のL2倍率を参照するが、微小ノルム時の扱いは下記の明示的なMOSS仕様とする。

## 確定するインターフェース・数値仕様

### 設定と入力

`TrainingConfig`の末尾に`teacher_normalization: str = "none"`を追加する。許可値以外、None、bool、数値はValueError。
既存の位置引数の意味を変えない。設定値は既存の`asdict(config)`経由で保存する。

YAMLは必須3キー＋任意`neutral`だけを許可する。
明示neutralは空白除去後に非空の文字列であること。`null`・空文字・空白のみ・数値は拒否する。
省略時にneutralキーをYAML snapshotへ補充しない。元の本文・キー構成・SHA256を維持する。
YAML anchor/alias、重複キー拒否、ファイルサイズ・レコード数制限は現行のまま。

手動入力は`neutral_conditioning=None`を省略扱いにし、非Noneなら既存の`normalize_conditioning`と同じ制限で検証する。
無効なneutralはnoneモードでも拒否する。未知キー・anchor等は引き続き拒否する。

### 教師と正規化

noneでは既存関数をそのまま呼ぶ:

```text
T = slider_teacher(base, positive, negative, eta)
```

neutral_l2では教師の組み立てからFP32で計算する:

```text
T = slider_teacher(base.float(), positive.float(), negative.float(), eta)
R = base(neutral)  # neutral省略時は既に計算したbase(target)を再使用
norm_T = L2(T)    # batch軸以外の全要素、sampleごと
norm_R = L2(R)
scale = norm_R / norm_T
T_normalized = T * scale
```

正規化するのは教師全体Tであり、`positive-negative`だけではない。
実装は`training.py`内の次の関数にまとめ、既存`slider_teacher`の署名を変更しない:

```python
def normalize_teacher_l2(
    teacher: Tensor, reference: Tensor
) -> tuple[Tensor, dict[str, Tensor]]:
    ...
```

- 入力は同じshape・deviceの浮動小数点tensor、次元数2以上、全要素が有限であること。異常入力はValueError。
- 出力はFP32、入力と同じshape/device、requires_grad=False。入力をin-place変更しない。
- `torch.no_grad()`で演算し、teacher/referenceの両方をdetachする。
- 固定閾値`1e-8`を使用する。いずれかのノルムが`<=1e-8`なら、そのsampleはscale=1で未正規化のTを返す。
- 分母clampだけで微小Tを増幅しない。ゼロreferenceを理由に教師全体をゼロへ潰さない。
- ノルム・scale・出力が非有限になった場合はRuntimeErrorとして停止する。正常域の倍率を恣意的に上限clipしない。
- statsは`teacher_l2`、`reference_l2`、`scale`、`skipped`の4キー。各値はshape `[B]`のdetach済みtensor、skippedだけbool。
- 参照と教師のノルムが等しい正常域ではscale=1、skipped=False。

この縮退時のskip処理は、Krea2の分母clamp処理との意図的な差である。
また、明示neutralがtargetと異なる場合、eta=0でもneutral_l2はbase(target)を再スケールし得る。
この点はテスト・説明に含め、eta=0を特別扱いして式を変更しない。noneでは従来のeta=0動作を維持する。

### 学習ループとレポート

- rolloutは従来どおりtarget条件のみ。学習レコード順・時刻／ノイズ抽出・student入力・optimizer更新数を変えない。
- noneではCPU上のneutralを検査しても、neutral用のGPU転送・教師forwardは行わない。
- neutral_l2で省略時は既存target予測を使うため、通常の3教師forwardのまま。
- neutral_l2で異なるneutralが明示された場合は、同じx・sigma・LoRA無効・no_gradで追加の教師forwardを1回行う。
- target/negative等の一般的な教師キャッシュ最適化は今回は行わず、noneの計算経路を保つ。
- 追加forwardの直前に既存cancel確認を置き、出力は既存の`_validate_prediction`を通す。

trainingレポートに`teacher_normalization`を追加:
- none: `{"mode": "none"}`。
- 有効: `{"mode": "neutral_l2", "epsilon": 1e-8, "reduction": "per_sample_non_batch_l2", "degenerate_policy": "skip_if_either_norm_le_epsilon", "dtype": "float32"}`。

有効時だけ各stepに`teacher_normalization_info`を追加する。
batch1契約に従い、`reference_source`は`target`または`neutral`、
`teacher_l2`・`reference_l2`・`scale`はPython float、`skipped`はboolとしてJSON化する。
tensor、NaN、Infinityをレポートへ入れない。noneの従来stepフィールドは変えない。
明示neutralの本文は既存`prompt_source.records`に保持する。手動条件に存在しない本文を捏造しない。

## Task 1: 数値関数と設定契約

**Files:** Modify `moss_slider_node/training.py`; Test `tests/unit/test_training.py`。

**Interfaces:** 上記`TrainingConfig.teacher_normalization`と`normalize_teacher_l2`を実装する。まだ学習ループからは有効化しない。

- [ ] Step 1: 変更前に`tests/unit/test_training.py`の`_toy(31)`と`_conditions()`を使用し、`TrainingConfig(steps=3, rank=2, alpha=2.0, learning_rate=0.01, width=16, height=16, rollout_steps=3, seed=7)`で実行する。全LoRA値と数値step情報を`test-results/neutral-l2/baseline-before.pt`へ保存する。時刻類を除外し、Python/PyTorch版と基準commitを併記する。同じ環境で変更後と比較するための一時的な検証資産とする。
- [ ] Step 2: `test_teacher_normalization_config_values`を追加し、既定none、neutral_l2受理、未知文字列／None／bool／数値拒否を検証する。既存default比較にも新しい既定fieldを明示する。
- [ ] Step 3: 数値テストを追加する。代表例:

```python
teacher = torch.tensor([[3., 4.], [0., 2.]], requires_grad=True)
reference = torch.tensor([[0., 10.], [0., 1.]], requires_grad=True)
out, stats = normalize_teacher_l2(teacher, reference)
torch.testing.assert_close(out, torch.tensor([[6., 8.], [0., 1.]]))
torch.testing.assert_close(stats["scale"], torch.tensor([2., 0.5]))
assert not out.requires_grad and not stats["skipped"].any()
```

- [ ] Step 4: `test_neutral_l2_skips_degenerate_norms`でT=0、R=0、両方0、ノルム1e-9、閾値ちょうど1e-8を確認する。outは元TのFP32値、scale=1、skipped=True。
- [ ] Step 5: `test_neutral_l2_validates_and_does_not_mutate_inputs`でshape不一致、整数入力、NaN/Infを拒否し、入力不変を確認する。FP32演算でノルムがoverflowする有限入力もRuntimeErrorになることを検証する。
- [ ] Step 6: `python -m pytest tests/unit/test_training.py -q`で新テストの失敗を確認後、最小実装を加え、同じコマンドでPASSにする。BF16入力からFP32出力になるテストも含める。
- [ ] Step 7: 差分を確認し、この独立した変更単位を記録する。推奨commit名: `feat: add optional neutral L2 teacher normalization primitive`。

## Task 2: 任意roleの受け渡し・学習接続・記録

**Files:** Modify `moss_slider_node/prompt_files.py`, `job.py`, `training.py`; Test `tests/unit/test_prompt_files.py`, `tests/unit/test_training.py`, `tests/integration/test_job.py`。

**Interfaces:** `load_prompt_file`、`encode_prompt_records`、`run_training_job`、`train_slider`の既存呼出し署名を維持する。
内部の条件dictは必須3role＋任意neutralを受理する。Task 1の設定と関数を使用する。

- [ ] Step 1: `test_optional_neutral_yaml_is_preserved_not_filled`で3キー／4キー／混在を受理し、省略レコードは3キーのまま、ファイルhashが原文由来であることを確認する。null・空白・未知キー・重複neutralは拒否する。
- [ ] Step 2: `test_encoder_optional_neutral_cache_and_progress`でroleの順をtarget/positive/negative/neutralとし、実際に存在するroleだけを処理する。3role＋4roleなら進捗total=7、同じ文章は既存cacheで1度だけencodeし、cancelを確認する。
- [ ] Step 3: jobの3キー完全一致検査を必須＋任意の検査へ変更する。`test_invalid_neutral_rejected_before_unload`で無効shape・mask・metadata・非有限値をモデルunload前に拒否する。neutralも既存`normalize_conditioning`へ渡す。
- [ ] Step 4: `_validate_conditions`でも必須／許可roleを明示し、neutralを既存と同じCPU・有限tensor条件で検査する。未知roleの黙殺をしない。
- [ ] Step 5: `test_none_ignores_valid_neutral_in_training`で、同じtoy初期化・seedについてneutralなし／ありのloss・grad norm・時刻index・noise index・最終全LoRA値が一致することを検証する。正規化関数を「呼ばれたら失敗」に置換してnone経路を固定する。
- [ ] Step 6: `test_neutral_l2_uses_target_fallback_or_explicit_prediction`で、省略時はtarget予測再利用、異なるneutral明示時はno_grad・multiplier0の追加forward、studentはtarget・multiplier1・1回だけであることを検証する。rollout数を差し引いた教師呼出し数も確認する。
- [ ] Step 7: `test_neutral_l2_matches_expected_teacher_and_reports`で既知のtoy予測から期待するL2教師とMSEを独立に計算し、loss・参照元・倍率・skip記録を照合する。eta0＋異なるneutralの再スケールも確認する。
- [ ] Step 8: 上記をRED→実装→GREENの順で進める。実装ではnone分岐を元の式のまま残し、有効分岐だけFP32教師生成・追加参照・正規化を行う。
- [ ] Step 9: jobの小型native検証で設定・任意neutral本文・step情報がmanifestに保存され、JSONを`allow_nan=False`で再シリアライズできることを確認する。既存の6YAML・各6件・3キーというテストは維持する。
- [ ] Step 10: `python -m pytest tests/unit/test_training.py tests/unit/test_prompt_files.py tests/integration/test_job.py -q`をPASSにする。推奨commit名: `feat: route optional neutral conditions through MOSS slider training`。

## Task 3: V3ノード・ワークフロー・利用説明

**Files:** Modify `nodes.py`, `README.md`, `prompts/README.md`, `workflows/moss_slider_train.json`, `moss_slider_train_api.json`, `moss_slider_train_prompts.json`, `moss_slider_train_prompts_api.json`; Test `tests/integration/test_nodes.py`。

**Interfaces:**
- 手動executeの既存引数の末尾へ`teacher_normalization="none", neutral_conditioning=None`を追加する。
- schema末尾へ`teacher_normalization`のCombo（options=none/neutral_l2、default=none、optional=True、advanced=True）と任意`neutral_conditioning`を追加する。
- YAMLノードは手動4条件の入力をすべて除外し、設定Comboのみ継承する。neutral本文はYAMLから受け取る。

- [ ] Step 1: `test_neutral_schema_is_optional_and_legacy_order_is_preserved`でV3→V1 metadataのoptional/defaultを確認する。既存input・widgetの相対順を変えない。YAMLノードにneutral_conditioning端子を出さない。
- [ ] Step 2: `test_old_api_omits_new_inputs_and_keeps_none`で旧execute呼出しと旧API JSONの入力が受理されることを検証する。Python既定引数だけでなくComfyのprompt入力検証も確認対象にする。
- [ ] Step 3: `test_manual_and_yaml_nodes_forward_normalization`で両ノードが同じ設定値を渡し、手動Noneはrole省略、接続時はneutralを渡すことを確認する。YAMLのエンコード進捗はencoder callbackの実totalを使い、固定の「件数×3」を除く。
- [ ] Step 4: schemaとexecuteを更新し、`python -m pytest tests/integration/test_nodes.py -q`をPASSにする。拡張登録の4ノードIDは増減させない。
- [ ] Step 5: 同梱API JSONに`teacher_normalization: "none"`を明示し、編集用JSONのwidgets_valuesは末尾にnoneを追加する。元の数値・prompt本文・接続は保つ。旧JSONはテストfixtureとして保持し、読み込み確認に使う。
- [ ] Step 6: 説明へ「noneは無効」「neutral省略はtarget参照」「L2は全教師を補正」「anchorではない」「微小ノルム時skip」「異なるneutralではeta0でも補正」を記載する。neutral追加損失という旧表現を、正規化と保持損失を区別する表現へ直す。
- [ ] Step 7: YAMLの例はREADME内に記載し、既存presetファイルは変更しない。neutralを省略したままneutral_l2を試せることも示す。推奨commit名: `feat: expose neutral normalization in MOSS training nodes`。

## Task 4: 互換性・失敗経路・実機の確認

**Files:** Extend `tests/integration/test_job.py`, `tests/integration/test_inference.py` as needed; Record results under `test-results/neutral-l2/`。原則としてexport/inferenceの製品コード変更は不要。

- [ ] Step 1: `test_cancel_during_neutral_teacher_restores_job_state`で追加forward直前のcancelを注入し、job lock・Comfy学習状態・LoRA multiplierを復旧できることを確認する。非有限neutral予測でも未完成LoRAを完成名で残さない。
- [ ] Step 2: `test_new_normalization_manifest_uses_existing_apply_format`で小型nativeモデルをneutral_l2で1～2step学習し、保存・読込・bypass適用を確認する。強度0の基底一致とINT8基底不変は既存テストを再利用する。
- [ ] Step 3: 新fieldのない旧manifestを読み込めること、新設定の有無でLoRAキー・shape・QKV共有Aの形式が変わらないことを確認する。旧artifactを新設定へ書き換えない。
- [ ] Step 4: Task 1の同一環境CPU基準を再実行し、noneの全LoRA値と数値step情報を変更前と照合する。時刻と追加された上位設定情報は比較対象外とする。
- [ ] Step 5: リポジトリ直下から以下を実行する。収集失敗や全skipをPASSとしない。既存`.test-deps`が必要ならその環境を明示して使用する。

```powershell
$env:MOSS_COMFYUI_ROOT = (Resolve-Path '..\ComfyUI').Path
python -m pytest tests/unit/test_training.py tests/unit/test_prompt_files.py tests/integration/test_job.py tests/integration/test_nodes.py tests/integration/test_inference.py -q
python tools/run_cpu_validation.py --comfyui-root "../ComfyUI" --report "test-results/neutral-l2/cpu-validation.json"
python -m ruff check .
python -m mypy -p moss_slider_node
git diff --check
```

- [ ] Step 6: 利用可能な実ComfyUIで旧学習workflowが正しく開き、追加widgetがnoneで、旧値がずれていないことを確認する。両学習ノードの`/object_info`とAPI入力検証も確認する。
- [ ] Step 7: 実機では同じ基底・seed・小さな設定でnone、neutral_l2＋省略、neutral_l2＋明示の短い学習・保存・Applyを確認する。GPUが利用できない場合はその実機確認だけを未検証と明記し、CPU検証と区別する。長時間学習・品質比較はここで自動実行しない。
- [ ] Step 8: 変更ファイル・テスト結果・実機確認の有無を報告する。推奨commit名: `test: verify neutral normalization compatibility and failure recovery`。

## 完了条件と実行方法

- 必須3条件のみの旧利用方法が動作し、noneで旧数値動作を維持できる。
- 任意neutral・targetへのfallback・L2通常域・縮退skipが仕様どおりで、レポートから区別できる。
- 有効時だけ追加の教師処理を行い、基底凍結・teacher detach・student +1を維持する。
- YAML、手動ノード、UI metadata、旧／新API、保存・Applyのテストが揃う。
- CPUテスト・静的検証を通過し、GPU／UI検証の実施範囲を正確に報告する。
- 実装成功を「服装の副作用が改善した」とは報告しない。品質比較は後続の同一条件A/Bで行う。

推奨実行方法は、同一担当がTask 1～4を順に実装し、完成差分を別担当がレビューする方式。
本書作成時点では実装を開始していない。

## 実装結果（2026-10-20）

Task 1～4の製品実装・CPU検証を完了。ユーザー変更指示に従い既定値はneutral_l2、全6YAML・36レコードにneutral=targetを追加した。
noneは変更前の固定CPU toyの全LoRA値・数値stepログと完全一致。元の3roleのプロンプト本文と既存workflow widgetの順序を維持した。
174 unit＋39 integration＝213テスト、ruff・mypy・構文・CPU境界・source integrityがPASS。実Comfyの旧API入力検証と、正規化学習後の保存・bypass Applyの往復もCPUで確認した。
独立コードレビューは指摘0件。検証レポートは`../../test-results/neutral-l2/cpu-validation.json`。
GPUおよびブラウザーでの実UI確認はCPU専用環境のため未実施。画像品質の改善は今後の比較対象。
