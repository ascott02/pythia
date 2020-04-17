#!/bin/bash
exp="ycII_ft_bs_64_lr_005"

python tools/run.py --tasks captioning --datasets youcookII \
    --model butd --config configs/captioning/youcookII/butd005.yml \
    --batch_size 64 -nw 32 --clip_gradients True \
    --resume_file content/model_data/butd.pth -pt 50000 \
    -exp ${exp} \
    --save_dir save/${exp} \
    -dev cuda:0 \
    # --distributed True --data_parallel True 
