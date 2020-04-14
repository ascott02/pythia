python build_imdb.py \
  --out_file /home/ats/code/pythia/content/youcookII_train_val_test_theshold_0.17/imdb_youcookII_validation.npy \
  -i /home/ats/code/pythia/content/youcookII_train_val_test_theshold_0.17/validation_features_resnet \
  -d /home/ats/sdb6/ats/youcook2/YouCookII/annotations/youcookii_annotations_trainval.json

python build_imdb.py \
  --out_file /home/ats/code/pythia/content/youcookII_train_val_test_theshold_0.17/imdb_youcookII_training.npy \
  -i /home/ats/code/pythia/content/youcookII_train_val_test_theshold_0.17/training_features_resnet \
  -d /home/ats/sdb6/ats/youcook2/YouCookII/annotations/youcookii_annotations_trainval.json

python build_imdb.py \
  --out_file /home/ats/code/pythia/content/youcookII_train_val_test_theshold_0.17/imdb_youcookII_test.npy \
  -i /home/ats/code/pythia/content/youcookII_train_val_test_theshold_0.17/test_features_resnet \
  -d /home/ats/sdb6/ats/youcook2/YouCookII/annotations/youcookii_annotations_trainval.json
