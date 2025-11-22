nohup python -u ../main.py \
    -data ../data/PharmKG/pro_data/ \
    -o ../pharm \
    --model_load_path ../klar/textual_encoder_checkpoint/PharmKG \
    --KG_embedding_path ../klar/graph_encoder_checkpoint/PharmKG/trained_999.pth \
    --max_path_len 6 \
    --freeze \
    -lr=1e-6 \
    -m \
    -me=16 \
    -mh=256 \
    -mk=2 \
    -b=64 \
    --metric_name Eval-hits10 \
    --data_loader_workers 10 \
    -e=15 \
   >log.log 2>&1 &
