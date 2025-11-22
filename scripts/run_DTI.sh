nohup python -u ../main.py \
    -data ../data/DTI/pro_data/ \
    -o ../DTI \
    --model_load_path ../klar/textual_encoder_checkpoint/DTI \
    --KG_embedding_path ../klar/graph_encoder_checkpoint/DTI/trained_999.pth \
    --max_path_len 6 \
    --freeze \
    -lr=1e-6 \
    -m \
    -me=64 \
    -mh=64 \
    -mk=2 \
    -b=64 \
    --metric_name Eval-hits10 \
    --data_loader_workers 10 \
    -e=30 \
   >log.log 2>&1 &
