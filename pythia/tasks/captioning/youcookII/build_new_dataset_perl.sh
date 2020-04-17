python build_new_dataset.py \
  -o ~/data/pythia/ \
  -t 0.15 -s 66,22,12 \
  -tfr ~/data/pythia/youcookII_features/training_features_resnet \
  -tfv ~/data/pythia/youcookII_features/training_features_vmb \
  -vfr ~/data/pythia/youcookII_features/validation_features_resnet \
  -vfv ~/data/pythia/youcookII_features/validation_features_vmb \
  -a ~/data/youcookII/YouCookII/annotations/youcookii_annotations_trainval.json \
  -tsf /mnt/sda1/youcookII/YouCookII/scripts/training_keyframe_scores_all.txt \
  -vsf /mnt/sda1/youcookII/YouCookII/scripts/training_keyframe_scores_all.txt

