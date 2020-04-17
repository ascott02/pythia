#!/bin/bash
exp="ycII_trn_bs_64_lr_01_no_preload"
python tools/run.py --tasks captioning --datasets youcookII \
    --model butd --config configs/captioning/youcookII/butd01.yml \
    --batch_size 64 -nw 32 --clip_gradients True \
    -exp ${exp} \
    --save_dir save/${exp} \
    -dev cuda:2 \
    # --resume_file content/model_data/butd.pth -pt 50000 \
    # --distributed True --data_parallel True 
