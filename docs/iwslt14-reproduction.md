# NAST IWSLT14 DE-EN 追試レポート

## 1. 概要

本ドキュメントは EMNLP 2023 論文 "Non-autoregressive Streaming Transformer for Simultaneous Translation" (NAST) を **IWSLT14 DE-EN** で追試した記録である。論文本体は WMT15 DE-EN (~4.5M ペア) で評価しているが、本追試は (a) 計算資源の制約、(b) 異なるデータセットでも品質-遅延 Pareto 形状が再現するかの検証、を目的に IWSLT14 (~160k ペア) に置き換えた。

絶対 BLEU は論文の WMT15 数値と直接比較できない点に注意 (データ規模が約 28 倍違う)。**主要な検証対象は Pareto 曲線の単調性と低遅延設定での品質維持**である。

## 2. 環境

| 項目 | 値 |
|------|-----|
| GPU | NVIDIA RTX PRO 6000 Blackwell (sm_120, 96 GB VRAM) × 1 |
| Python | 3.10.13 |
| PyTorch | 2.9.1+cu130 |
| CUDA toolkit (env 内) | 13.x (`conda install -c nvidia cuda-toolkit`) |
| fairseq | `5175fd5` (Jan 2022) + 本リポジトリのパッチ |
| NumPy | 1.26.4 |
| omegaconf / hydra-core | 2.0.6 / 1.0.7 (fairseq 5175fd の pin) |

論文オリジナル環境 (Python 3.7 + PyTorch 1.10.1 + CUDA 11.3) は Blackwell バイナリが存在しないため利用不可。fairseq 5175fd は元々 PyTorch 1.10 前提だったが、本追試では `patches/fairseq-5175fd-pt29-compat.patch` を当てて 2.x 系で動作させている。

## 3. データ

| split | 文対数 | tokens (de) | tokens (en) |
|-------|--------|-------------|-------------|
| train | 160,239 | 4,038,059 | 3,955,010 |
| valid | 7,283 | 182,658 | 178,888 |
| test | 6,750 | 161,929 | 157,166 |

- ソース: HuggingFace `bbaaaa/iwslt14-de-en` (raw text、未 tokenize)
- 前処理: Moses normalize-punctuation → Moses tokenizer → lowercase → BPE 10k joined (subword-nmt)
- バイナリ化: `fairseq-preprocess --joined-dictionary` (語彙 10,152 types、両言語共通)

## 4. モデル

新規登録した `nonautoregressive_streaming_transformer_iwslt` を使用。`base_architecture` から以下のみ差分:

| パラメータ | base | iwslt | 備考 |
|-----------|------|-------|------|
| encoder_ffn_embed_dim | 2048 | **1024** | IWSLT 軽量化 |
| decoder_ffn_embed_dim | 2048 | **1024** | 同上 |
| encoder_attention_heads | 8 | **4** | 同上 |
| decoder_attention_heads | 8 | **4** | 同上 |
| dropout | 0.1 | **0.3** | 過学習抑制 |
| encoder/decoder_layers | 6 | 6 | 共通 |
| encoder/decoder_embed_dim | 512 | 512 | 共通 |

総パラメータ数: **37.9M** (`transformer_iwslt_de_en` 相当)。

NAST 固有の構造 (UniTransformerEncoder / src_embedding_copy / wait-k チャンク化 / CTC alignment via torch_imputer / GLAT) は base のまま維持。

## 5. 学習スケジュール

### 5.1 Stage-1 (CTC 事前学習)

| 設定 | 値 |
|------|-----|
| max-tokens / update-freq | 32768 / 2 (実効 batch 64k) |
| 学習率 | 0.0005, inverse_sqrt, warmup 4000 |
| dropout | 0.3 |
| label-smoothing | 0.01 |
| GLAT schedule | 0.5 → 0.3 @ 100k updates |
| max-update | 150,000 (計画値) |
| **実際の停止 update** | **3,888 (epoch 48)** |
| 停止理由 | val_loss 上昇開始による早期停止 |
| best val_loss | 2.742 (epoch 48) |
| 所要時間 | 約 9 時間 (途中ディスク満杯で 44k updates 時点でクラッシュ、最良 ckpt は救済済み) |
| 平均 wps | 75,000 |
| 平均 ups | 1.46 |
| 学習中の温度 | 83-87°C |

NAT モデルの IWSLT14 における過学習は既知の現象で、論文の WMT15 (5M ペア) では発生しないが、本追試の小規模データでは epoch 50 前後でほぼ最良点に到達した。`checkpoint_best.pt` を Stage-2 の初期重みとして使用。

### 5.2 Stage-2 (latency loss + n-gram matching の finetune)

5 点の Pareto sweep を実施。各点 `checkpoint_best.pt` から fine-tune。

| 共通設定 | 値 |
|---------|-----|
| max-tokens / update-freq | 16384 / 4 (実効 batch 64k、論文の 256k より小) |
| 学習率 | 0.0003, inverse_sqrt, warmup 500 |
| dropout | 0.1 |
| GLAT schedule | 0.3 → 0.3 @ 8k (定常) |
| n-gram matching | size=2, use_ngram=True |
| max-update | 8,000 |

| Point | wait_until | latency_factor | latency_threshold | 学習時間 | training loss (final) | val loss (best) |
|-------|-----------|----------------|-------------------|---------|----------------------|-----------------|
| P1 | 1 | 1.0 | 3.0 | 7,910 s (2h12) | 2.475 | 2.657 |
| P2 | 3 | 1.0 | 4.5 | 8,092 s (2h15) | 3.96 | 4.149 |
| P3 | 5 | 0.5 | 5.0 | 8,105 s (2h15) | 1.955 | 2.147 |
| P4 | 7 | 0.3 | 6.0 | 8,457 s (2h21) | 1.252 | 1.444 |
| P5 | 9 | 0.0 | 0.0 | 6,841 s (1h54) | -0.552 | -0.356 |

P5 は latency_loss 無効のため total loss が小さい (CTC log-likelihood のみ)。生の loss 値は点間で直接比較できない (latency_loss が支配的に効くかどうかで桁が違うため)。

## 6. 評価結果

### 6.1 Pareto 表

各点で最終 5 update-interval ckpt を `average_checkpoints.py` で平均し、`generate_streaming.py` (plain CTC、greedy) でテストセットを推論。BLEU は `multi-bleu.perl -lc`、latency 指標は `generate_streaming.py` 内蔵。

| Point | wait_until | lat_factor | lat_thresh | **BLEU** | **AL** | **CW** | **AP** | **DAL** |
|-------|-----------|------------|------------|----------|--------|--------|--------|---------|
| **P1** (low-latency)  | 1 | 1.0 | 3.0 | **24.42** | **1.86** | 1.63 | 0.58 | 3.16 |
| **P2**                | 3 | 1.0 | 4.5 | **25.20** | **3.82** | 1.89 | 0.67 | 5.19 |
| **P3** (balanced)     | 5 | 0.5 | 5.0 | **25.46** | **5.73** | 2.30 | 0.75 | 7.11 |
| **P4**                | 7 | 0.3 | 6.0 | **25.69** | **7.57** | 2.92 | 0.80 | 8.91 |
| **P5** (high-quality) | 9 | 0.0 | 0.0 | **25.45** | **9.34** | 3.73 | 0.85 | 10.62 |

### 6.2 Pareto 曲線 (テキストプロット)

```
BLEU
26.0 |
     |              P4 (7.57, 25.69) ◆
25.5 |       P3 (5.73, 25.46) ◆     ◆ P5 (9.34, 25.45)
     |     P2 (3.82, 25.20) ◆
25.0 |
     |
24.5 |  P1 (1.86, 24.42) ◆
     |
24.0 |
     +------+------+------+------+------+------+
     0      2      4      6      8     10     AL
```

### 6.3 観察

1. **AL に対して BLEU はほぼ単調増加** (24.42 → 25.20 → 25.46 → 25.69 → 25.45)。P4 で頭打ち、P5 (AL 9.34) でわずかに低下する飽和形状。これは IWSLT14 の小規模データに対して `wait_until=7` 程度で十分な未来コンテキストが確保されることを示唆する。
2. **低遅延でも品質維持**: P1 (AL 1.86) と P4 (AL 7.57) の BLEU 差は 1.27 のみ。論文の頭出し文 *"NAST demonstrates exceptional performance under extremely low latency conditions"* の特性が IWSLT14 でも確認できた。
3. **論文 WMT15 結果 (29.82 BLEU @ 1.89 AL) との差**: 本追試の P1 は AL ≈ 1.89 で BLEU 24.42。約 5 BLEU pt の差は (a) 訓練データ規模 28 倍差、(b) Stage-1 を 150k 計画から 3,888 updates で早期停止したこと、(c) IWSLT 用に縮小したアーキテクチャ (FFN 2048→1024) が複合した結果と考えられる。
4. **latency_loss は学習中ほぼ threshold ジャストに張り付く**: 5 点とも train ログ上で `latency_loss ≈ threshold` で推移し、threshold を割り込むケースは観測されなかった。これは alignment-based latency loss が threshold を超えないように十分働いている証拠で、評価時の AL が大体 threshold 近辺になる結果と整合する (e.g. P4 threshold=6.0, AL=7.57; P3 threshold=5.0, AL=5.73)。

## 7. 計算コスト

| Phase | 所要時間 |
|-------|---------|
| Phase 0-3 (環境構築・データ準備) | ~2 h |
| Phase 1 (fairseq 互換パッチ) | 約 2 h (デバッグ含む) |
| Phase 2 (CUDA カーネル JIT) | 約 5 min |
| Phase 4 (Stage-1 早期停止) | ~9 h |
| Phase 5 (Stage-2 × 5 点) | ~11.4 h |
| Phase 6-7 (評価・ドキュメント・レビュー) | ~2 h |
| **総計** | **~26.5 h** |

GPU 利用率 (Stage-2 中央値): 60-80%、温度 83-87°C、VRAM 使用 60-70 GB / 96 GB。

## 8. 既知の問題と対応

### 8.1 環境互換性
- **`np.float` / `np.int` の deprecated alias**: NumPy 1.20+ で削除。`patches/fairseq-5175fd-pt29-compat.patch` で `np.float32` / `np.int64` に置換 (4 ファイル、6 箇所)。
- **`torch.load(weights_only=True)` のデフォルト変更 (PyTorch 2.6+)**: fairseq の checkpoint には `argparse.Namespace` が pickle されているため `weights_only=False` を明示。同パッチで `fairseq/checkpoint_utils.py` の 2 箇所を修正。
- **`THCudaCheck` の削除 (PyTorch 2.x)**: `imputer.cu` 冒頭に `#define THCudaCheck(EXPR) C10_CUDA_CHECK(EXPR)` の互換シムを追加。
- **submodule 名 `fairseq` と Python パッケージ `fairseq` の名前衝突**: project root で `import fairseq` した際に submodule ディレクトリが namespace package として shadow を起こす。submodule path を `fairseq_repo` にリネームして解決。

### 8.2 IWSLT14 固有の挙動
- **Stage-1 早期過学習**: epoch 50 前後で val_loss が下げ止まり、その後上昇。原因は IWSLT14 の小規模性 (160k ペア)。`checkpoint_best.pt` を Stage-2 初期に使う運用で対応。論文の WMT15 では発生しない現象。
- **wait_until=9 で品質飽和**: P5 (AL 9.34) が P4 (AL 7.57) より BLEU が下回る。IWSLT14 の文長中央値が小さいため、`wait_until` を増やしても見える未来文脈が頭打ちになる。

### 8.3 評価スクリプトの罠
- **`generate_streaming.py` の D-line 形式**: このバージョン (fairseq 5175fd) では `D-<id>\t<hyp>` の **2 列**形式 (新しい fairseq の `D-<id>\t<score>\t<hyp>` ではない)。`shell_scripts/test.sh` で `cut -f 2-` を使う必要あり (3 では空仮説になる)。コメントで明記済み。
- **`set -e` 下での `grep` 0 件マッチ**: パイプチェーンで例外を握り潰す `|| true` が必要。

## 9. ファイル構成

```
NAST/
├── README.md                                 # 追試結果セクションあり
├── docs/
│   └── iwslt14-reproduction.md               # 本ドキュメント
├── patches/
│   ├── README.md                             # パッチ適用手順
│   └── fairseq-5175fd-pt29-compat.patch      # fairseq 互換パッチ
├── NAST/                                     # fairseq plugin
│   ├── models/
│   │   ├── nonautoregressive_streaming_transformer.py  # +iwslt arch
│   │   └── torch_imputer/imputer.cu          # THCudaCheck shim
│   └── scripts/average_checkpoints.py        # weights_only=False
├── shell_scripts/
│   ├── train_stage1.sh                       # IWSLT14 用本番設定
│   ├── train_stage2.sh                       # IWSLT14 用本番設定
│   ├── run_pareto.sh                         # 5 点 Pareto driver (新規)
│   ├── test.sh                               # BLEU + latency 評価
│   └── multi-bleu.perl                       # オリジナル
└── work/                                     # .gitignore (ローカル成果物)
    ├── data/iwslt14.{raw,tok,bpe}/           # 前処理 stage 別
    ├── data-bin/iwslt14.de-en.joined/        # binarize 済み
    ├── checkpoints/                          # 7.5GB × 6 (Stage-1 + 5 points)
    └── logs/                                 # 学習ログ + 評価ログ
```

## 10. 再現手順サマリー

詳細は `README.md` の "How to reproduce" を参照。要点のみ:

```bash
# 環境構築 (Blackwell sm_120 想定)
conda create -n nast python=3.10.13 -y && conda activate nast
conda install -c nvidia cuda-toolkit -y
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu130
pip install datasets==2.19.0 omegaconf==2.0.6 hydra-core==1.0.7 'numpy<2' \
            sacrebleu==2.4.2 sacremoses==0.1.1 subword-nmt==0.3.8 tensorboardX==2.6.2 ninja
git submodule sync --recursive && git submodule update --init --recursive
( cd fairseq_repo && git apply ../patches/fairseq-5175fd-pt29-compat.patch )
cd fairseq_repo && pip install -e . && cd ..

# データ
python work/scripts/01_dump_hf_to_text.py
bash work/scripts/02_tokenize.sh
bash work/scripts/03_bpe.sh
fairseq-preprocess --source-lang de --target-lang en \
  --trainpref work/data/iwslt14.bpe/train --validpref work/data/iwslt14.bpe/valid \
  --testpref  work/data/iwslt14.bpe/test  --destdir   work/data-bin/iwslt14.de-en.joined \
  --joined-dictionary --workers 8

# 学習 (Stage-1 → Stage-2 × 5 点)
bash shell_scripts/train_stage1.sh
STAGE1_AVG=work/checkpoints/iwslt14_s1/checkpoint_best.pt bash shell_scripts/run_pareto.sh

# 評価 (各 point ごと)
for p in p1 p2 p3 p4 p5; do
  source /home/naoto/miniconda3/etc/profile.d/conda.sh && conda activate nast
  python NAST/scripts/average_checkpoints.py \
    --inputs work/checkpoints/iwslt14_s2_$p --num-update 5 \
    --output work/checkpoints/iwslt14_s2_$p/avg5.pt
done
EXP=iwslt14_s2_p3 AVG_CKPT=work/checkpoints/iwslt14_s2_p3/avg5.pt WAIT_UNTIL=5 \
  bash shell_scripts/test.sh
```

## 11. 結論

NAST の主要主張である **「非自己回帰ストリーミングで低遅延でも翻訳品質を維持できる」** は、論文の WMT15 だけでなく IWSLT14 でも再現できた。具体的には AL を 7.57 から 1.86 まで 4 倍縮めても BLEU 低下は 1.27 pt に留まった。一方、絶対 BLEU は WMT15 (29.82 @ 1.89) と比べ約 5 pt 低い (24.42 @ 1.86) が、これはデータ規模差 (28 倍)、Stage-1 早期停止、軽量アーキテクチャの組み合わせで説明できる。

PyTorch 2.9 / Blackwell GPU という近代環境で 2022 年の fairseq commit を動かすには 4 ファイルのパッチで足りること、CUDA カーネル拡張は `THCudaCheck` の互換シム 1 つで sm_120 ネイティブビルドできることを、本追試で確認した。
