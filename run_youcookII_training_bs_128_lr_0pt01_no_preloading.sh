#!/bin/bash
# exp="ycII_ft_bs_64_lr_01"
exp="ycII_training_bs_128_lr_01_no_preload"
python tools/run.py --tasks captioning --datasets youcookII \
    --model butd --config configs/captioning/youcookII/butd01_no_preload.yml \
    --batch_size 128 -nw 32 --clip_gradients True \
    -exp ${exp} \
    --save_dir save/${exp} \
    -dev cuda:0 \
    # --distributed True --data_parallel True 
    # --resume_file content/model_data/butd.pth -pt 50000 \
