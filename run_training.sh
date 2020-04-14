time python tools/run.py --tasks captioning --datasets youcookII \
    --model butd --config configs/captioning/youcookII/butd.yml \
    --batch_size 64 -nw 32 --clip_gradients True \
    --resume_file content/model_data/butd.pth \
    -exp youcookII_fine-tune_bs_64_lr_0pt01 \
    --distributed True --data_parallel True \
