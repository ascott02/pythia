
# usage: extract_features_vmb.py [-h] [--model_file MODEL_FILE]
#                                [--config_file CONFIG_FILE]
#                                [--batch_size BATCH_SIZE]
#                                [--num_features NUM_FEATURES]
#                                [--output_folder OUTPUT_FOLDER]
#                                [--image_dir IMAGE_DIR]
#                                [--feature_name FEATURE_NAME]
#                                [--confidence_threshold CONFIDENCE_THRESHOLD]
#                                [--background]
# 
# optional arguments:
#   -h, --help            show this help message and exit
#   --model_file MODEL_FILE
#                         Detectron model file
#   --config_file CONFIG_FILE
#                         Detectron config file
#   --batch_size BATCH_SIZE
#                         Batch size
#   --num_features NUM_FEATURES
#                         Number of features to extract.
#   --output_folder OUTPUT_FOLDER
#                         Output folder
#   --image_dir IMAGE_DIR
#                         Image directory or file
#   --feature_name FEATURE_NAME
#                         The name of the feature to extract
#   --confidence_threshold CONFIDENCE_THRESHOLD
#                         Threshold of detection confidence above which boxes
#                         will be selected
#   --background          The model will output predictions for the background

# From old instructions, no longer using pkl format
#  --model_file /home/ascott/data/pythia/feature_extraction/FAST_RCNN_MLP_DIM2048_FPN_DIM512.pkl \
#  --config_file /home/ascott/data/pythia/feature_extraction/e2e_faster_rcnn_X-101-64x4d-FPN_1x_MLP_2048_FPN_512.yaml \

# From extract_features_vmb.py file
#   --feature_name gpu_0/fc6 \
#   --confidence_threshold .5 \
#   --num_features 6 \
#   --batch_size 8 \
#
#   --output_folder /home/ascott/data/pythia/feature_extraction/validation_copy_features \
#   --image_dir /home/ascott/data/youcookII/YouCookII/keyframes/validation_copy_flat \
python extract_features_vmb.py \
  --model_file /mnt/sdg1/pythia/content/model_data/detectron_model.pth \
  --config_file /mnt/sdg1/pythia/content/model_data/detectron_model.yaml \
  --output_folder /home/ascott/data/pythia/feature_extraction/training_copy_features \
  --image_dir /home/ascott/data/youcookII/YouCookII/keyframes/training_copy_flat \
  --background

