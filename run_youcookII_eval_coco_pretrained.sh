#!/bin/bash
exp="ycII_evl_coco_preload_bs_64_lr_01"
python tools/run.py --tasks captioning --datasets youcookII \
    --run_type validation \
    --model butd --config configs/captioning/youcookII/butd01.yml \
    --batch_size 64 -nw 32 --clip_gradients True \
    --resume_file content/model_data/butd.pth -pt 50000 \
    -exp ${exp} \
    --save_dir save/${exp} \
    -dev cuda:1 \
    # --distributed True --data_parallel True 
