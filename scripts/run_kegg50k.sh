#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

echo "=== [1/3] Preprocessing kegg50k & Resolving Biological Names from KEGG REST API ==="
python data/kegg50k/kegg50k_data_processing.py --num_walks 16 --max_walk_len 6 --seed 42

echo "=== [2/3] Pre-computing Frozen Textual Representations (Mean Pooling) ==="
for ENC in pubmedbert bert flant5; do
    echo "--> Extracting embeddings with ${ENC}..."
    python klar/data/precompute_text_emb.py \
        --pro_data_dir ./data/kegg50k/pro_data \
        --encoder ${ENC} \
        --batch_size 128 \
        --max_length 512
done

echo "=== [3/3] Training & Evaluating KLaR on kegg50k (3 Random Seeds on RTX 5090) ==="
for ENC in pubmedbert bert flant5; do
    echo "--> Training KLaR with frozen encoder: ${ENC}..."
    python train_klar.py \
        --pro_data_dir ./data/kegg50k/pro_data \
        --encoder ${ENC} \
        --d_s 256 \
        --d 256 \
        --num_rgnn_layers 2 \
        --num_experts 64 \
        --top_k 4 \
        --num_negatives 128 \
        --batch_size 1024 \
        --max_epochs 30 \
        --seeds 42 100 2026
done
