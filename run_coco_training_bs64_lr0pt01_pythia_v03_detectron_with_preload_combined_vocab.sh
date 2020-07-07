#!/bin/bash

# exp="coco_from_scratch_default_vocab_bs_192_lr_01_pythia_v0.4_run_28_detectron_dataparallel"
# python -m torch.distributed.launch --nproc_per_node 3 tools/run.py --tasks captioning --datasets coco \
exp="coco_training_bs64_lr0pt01_pythia_v03_detectron_with_preload_combined_vocab"
python tools/run.py --tasks captioning --datasets coco \
    --model butd --config configs/captioning/coco/butd.yml \
    --batch_size 64 -nw 18 --clip_gradients True \
    -exp ${exp} \
    --save_dir save/${exp} \
    # --data_parallel True \
    # --run_type inference
    # --resume_file save/coco_train_bs192_lr01_pythia_v04_detectron_dataparallel_ocr/coco_butd/butd_final.pth \

    # -dev cuda:2 \
    # training_parameters.pin_memory True 
    # --distributed True \
    # -lr 0
    # -dev cuda:0 \

