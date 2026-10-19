# 実機検証 — 2026-10-19

指定された image_moss_image_t2i.json のモデルで、起動中の ComfyUI を REST API 経由で検証した。**通常生成は成功。Slider 学習は checkpoint 形式の非互換により開始前に停止した。P8 の学習検証は未完了。**

## 環境

- サーバー: http://localhost:8000 / ComfyUI 0.42.0
- GPU: AMD Radeon AI PRO R9700、VRAM 約31.86 GiB（system_stats の値）
- Python 3.12.11 / torch 2.12.0+rocm7.14.0 / comfy-kitchen 0.2.39 / comfy-aimdo 0.5.6
- 稼働中ノードの python_module と traceback から、このリポの nodes.py が実行されたことを確認。
- 基底: moodyMossMixFreeDownload_v1025STEPSCFG3_int8.safetensors
- Text encoder: qwen3.5_9b_int8_convrot.safetensors、type=moss_image
- VAE: flux2-vae_small.safetensors

## 実行結果

| 項目 | 結果 | 実行ID |
|---|---|---|
| 小型 baseline: 512×512 / 12 step / 固定 seed | PASS、人物画像の出力・目視確認 | c6c6f57b-da3c-44ec-9515-19a8da86deae |
| MossSliderInspect | INVALID、schema 不一致 | baseline と同じ |
| MossSliderTrain: rank2 / alpha2 / 256×256 / 1 step / rollout4 / checkpointing有効 / swap0 | ERROR、job.py:50 で拒否、更新0回 | 5bde3a28-3744-4c13-8d4f-8a7d9c99bbb9 |
| 学習失敗後に元 workflow を実行: 1024×1024 / 25 step | PASS、TextGenerate→encode→sampler→VAE→SaveImage が完了 | f988f033-5648-4f90-9720-7bac0cb68b5b |

最初の baseline は portrait の直接入力に変更し、TextGenerate と SLG を除去、解像度512・step12へ縮小した。モデル3点、seed=333599710072681、er_sde/simple、CFG2.5 は指定 JSON と同じ。最後の実行は保存先 prefix のみ変更し、指定 JSON の生成構成・prompt・seed・1024解像度・25 step を保持した。履歴の execution_start→execution_success は約34.62秒（単一実行の総時間、学習速度ではない）。

## 根本原因

指定 checkpoint は 5,088,901,904 bytes、562 tensors（I8=184、BF16=194、F32=184）。StarNodes Model Converter の int8_convrot 形式で、184層の設定が __metadata__._quantization_metadata.layers にある。tensor としての .comfy_quant 設定および Attention 設定は存在しない。全184層の metadata は int8_tensorwise / convrot=true / groupsize=256。

現在の checkpoint.py:169 は layers.N.proj の4投影×42 block=168層だけを量子化対象として固定し、追加の scale を checkpoint.py:178 で拒否する。さらに checkpoint.py:193 は各投影の .comfy_quant tensor、続く検査は各 block の Attention 設定を必要とする。今回の重みは text_embedder、time_embedding、modulation、final_linear_video など追加16層も量子化している。このため設定の読み替えだけでは足りず、学習 loader の非対象層の扱いも検証が必要。

実際の例外:

~~~text
MOSS schema keys mismatch: missing=[], unexpected=['final_linear_video.weight_scale', 'shared_block_modulation.weight_scale', 'text_embedder.blocks.0.ffn_down.weight_scale', 'text_embedder.blocks.0.ffn_gate.weight_scale', 'text_embedder.blocks.0.ffn_up.weight_scale']
~~~

job.py:50 は unload_all_models、基底hash計算、学習モデル割当より前。したがって GPU backward、finite loss/gradient、LoRA export、Apply の ±/0、10/100 step、staging、peak VRAM、cancel/retry、学習品質は NOT_RUN。今回の結果は GPU OOM や ROCm backward 不具合の証拠ではない。checkpoint の形式検査を回避する変更は行っていない。

## 保存した証拠

[test-results/live-20261019](../test-results/live-20261019/) に system-stats、preflight、3実行の API JSON・submit・terminal history、checkpoint 検査・量子化内訳、ログ、生成画像2枚、生成された展開prompt、summary.json を保存。

- [人物 baseline](../test-results/live-20261019/baseline.png)
- [指定 workflow の出力](../test-results/live-20261019/original-workflow.png)
- [機械可読 summary](../test-results/live-20261019/summary.json)
- [実学習失敗の history](../test-results/live-20261019/train-smoke-history.json)

製品コードと指定元 JSON は変更していない。CPU133件の通過は .omx の以前の記録であり、今回再実行した結果ではない。ブラウザー上の編集・描画操作は検証対象に含めていない。

## 次の実装対象

metadata 由来の量子化形式の正規化、追加16層を凍結したままロード・計算する経路、Attention 設定を持たない形式の明示的な扱いを、小型回帰テスト付きで追加する。その後、同じ指定モデルで rank2・1 step から GPU 検証を再開する。
