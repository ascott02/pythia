import argparse
import glob
import json
import os
import numpy as np
import sys
import random
from shutil import copyfile

# from pythia.utils.preprocessing import text_tokenize
from pythia.utils.text_utils import tokenize

# annotations_file = "/home/ascott/data/youcookII/YouCookII/annotations/youcookii_annotations_trainval.json"
# fcnn_features_dir = "/home/ascott/data/pythia/youcookII_features/training_features_vmb"
# resnet_features_dir = "/home/ascott/data/pythia/youcookII_features/training_features_resnet"

# # training
# python build_imdb.py --out_file imdb_youcookII_training.npy -i /home/ascott/data/pythia/youcookII_features/training_features_resnet -d /home/ascott/data/youcookII/YouCookII/annotations/youcookii_annotations_trainval.json
# # validation
# python build_imdb.py --out_file imdb_youcookII_validation.npy -i /home/ascott/data/pythia/youcookII_features/validation_features_resnet -d /home/ascott/data/youcookII/YouCookII/annotations/youcookii_annotations_trainval.json

# builds new youcookII dataset splits, from oroginal training and validation segment keyframes
#
#   - input:  split percentations: default: 66, 22, 12
#             a threshold (float), resnet feature dir for training, resnet feature dir for validation, faster-rcnn feature dir for training, faster-rcnn feature dir for validation, 
#             annotations file for training and annotations
#             target_dir=/home/ascott/data/pythia/ 
#   - output: directories for training_youcookII_threshold_<threshold>, validation_youcookII_threshold_<threshold>, testing_youcookII_threshold_<threshold>,
#             and imdb files: imdb_youcookII_training_threshold_<threshold>, imdb_youcookII_validation_threshold_<threshold>, testing_youcookII_threshold_<threshold>
#             results in new dir, target_dir like: youcookII_train_val_test_threshold_<threshold>

# run with:
# python build_new_dataset.py 
#     -o /home/ascott/data/pythia/ \
#     -t 0.15 \
#     -s 66,22,12 \
#     -tfr /home/ascott/data/pythia/youcookII_features/training_features_resnet \
#     -tfv /home/ascott/data/pythia/youcookII_features/training_features_vmb \
#     -vfr /home/ascott/data/pythia/youcookII_features/validation_features_resnet \
#     -vfv /home/ascott/data/pythia/youcookII_features/validation_features_vmb \
#     -a /home/ascott/data/youcookII/YouCookII/annotations/youcookii_annotations_trainval.json \
#     -tsf /mnt/sda1/youcookII/YouCookII/scripts/training_keyframe_scores_all.txt \
#     -vsf /mnt/sda1/youcookII/YouCookII/scripts/training_keyframe_scores_all.txt
# python build_new_dataset.py -o /home/ascott/data/pythia/ -t 0.15 -s 66,22,12 -tfr /home/ascott/data/pythia/youcookII_features/training_features_resnet -tfv /home/ascott/data/pythia/youcookII_features/training_features_vmb -vfr /home/ascott/data/pythia/youcookII_features/validation_features_resnet -vfv /home/ascott/data/pythia/youcookII_features/validation_features_vmb -a /home/ascott/data/youcookII/YouCookII/annotations/youcookii_annotations_trainval.json -tsf /mnt/sda1/youcookII/YouCookII/scripts/training_keyframe_scores_all.txt -vsf /mnt/sda1/youcookII/YouCookII/scripts/training_keyframe_scores_all.txt
#
# score files:
# /mnt/sda1/youcookII/YouCookII/scripts/training_keyframe_scores_all.txt
# /mnt/sda1/youcookII/YouCookII/scripts/validation_keyframe_scores_all.txt
class SplitBuilder:
    def __init__(self):
        self.args = self.get_args()

    def get_args(self):
        parser = argparse.ArgumentParser("Build IMDB for YouCookII")
        parser.add_argument(
            "-t",
            "--threshold",
            type=float,
            default=0.15,
            help="Drop samples with cosine similarity score below this value",
        )
        parser.add_argument(
            "-o",
            "--out_dir",
            type=str,
            default="/home/ascott/data/pythia/",
            help="Output directory for new split dirs and IMDB files",
        )
        parser.add_argument(
            "-s",
            "--split_percentages",
            type=str,
            default="66,22,12",
            help="Percentage of dataset for each split (train,val,test)",
        )
        parser.add_argument(
            "-tfr",
            "--training_features_resnet",
            type=str,
            default="/home/ascott/data/pythia/youcookII_features/training_features_resnet",
            help="Path to directory of training resnet features",
        )
        parser.add_argument(
            "-tfv",
            "--training_features_vmb",
            type=str,
            default="/home/ascott/data/pythia/youcookII_features/training_features_vmb",
            help="Path to directory of training faster rcnn features",
        )
        parser.add_argument(
            "-vfr",
            "--validation_features_resnet",
            type=str,
            default="/home/ascott/data/pythia/youcookII_features/validation_features_resnet",
            help="Path to directory of validationresnet features",
        )
        parser.add_argument(
            "-vfv",
            "--validation_features_vmb",
            type=str,
            default="/home/ascott/data/pythia/youcookII_features/validation_features_vmb",
            help="Path to directory of validationfaster rcnn features",
        )
        parser.add_argument(
            "-tsf",
            "--training_scores_file",
            type=str,
            default="/mnt/sda1/youcookII/YouCookII/scripts/training_keyframe_scores_all.txt",
            help="Path to file containing cosine sim scores for training files",
        )
        parser.add_argument(
            "-vsf",
            "--validation_scores_file",
            type=str,
            default="/mnt/sda1/youcookII/YouCookII/scripts/validation_keyframe_scores_all.txt",
            help="Path to file containing cosine sim scores for validation files",
        )
        parser.add_argument(
            "-a",
            "--annotations_file",
            type=str,
            default="/home/ascott/data/youcookII/YouCookII/annotations/youcookii_annotations_trainval.json",
            help="JSON File with Annotations for YouCookII training and validation sets",
        )
        return parser.parse_args()

    def split_data(self):
        # annotations_file = self.args.annotations_file

        training_resnet_dir = self.args.training_features_resnet
        training_vmb_dir = self.args.training_features_vmb

        validation_resnet_dir = self.args.validation_features_resnet
        validation_vmb_dir = self.args.validation_features_vmb

        training_scores_file = self.args.training_scores_file
        validation_scores_file = self.args.validation_scores_file

        split_percentages = self.args.split_percentages

        threshold = self.args.threshold
        out_dir = self.args.out_dir

        splits = [float(x) for x in split_percentages.split(",")]

        target_out_dir = out_dir + "/youcookII_train_val_test_theshold_" + str(threshold)

        training_out_dir_resnet = target_out_dir + "/training_features_resnet"
        training_out_dir_vmb = target_out_dir + "/training_features_vmb"

        validation_out_dir_resnet = target_out_dir + "/validation_features_resnet"
        validation_out_dir_vmb = target_out_dir + "/validation_features_vmb"

        test_out_dir_resnet = target_out_dir + "/test_features_resnet"
        test_out_dir_vmb = target_out_dir + "/test_features_vmb"

        if not os.path.isdir(target_out_dir):
            os.mkdir(target_out_dir)

        if not os.path.isdir(training_out_dir_resnet):
            os.mkdir(training_out_dir_resnet)
        if not os.path.isdir(training_out_dir_vmb):
            os.mkdir(training_out_dir_vmb)

        if not os.path.isdir(validation_out_dir_resnet):
            os.mkdir(validation_out_dir_resnet)
        if not os.path.isdir(validation_out_dir_vmb):
            os.mkdir(validation_out_dir_vmb)

        if not os.path.isdir(test_out_dir_resnet):
            os.mkdir(test_out_dir_resnet)
        if not os.path.isdir(test_out_dir_vmb):
            os.mkdir(test_out_dir_vmb)
 
        # data = None
        # with open(annotations_file, "r") as f:
        #     data = json.load(f)

        fh = open(validation_scores_file, 'r')
        lines = fh.readlines()
        fh.close()

        fh = open(training_scores_file, 'r')
        lines += fh.readlines()
        fh.close()

        samples = []
        scores = {}
        # -0.045630150334090784 /home/ascott/data/youcookII/YouCookII/keyframes/validation/HdVETeyupXE/7/frame5450.jpg sprinkle saffron water on top
        for line in lines:
            line = line.rstrip()
            line = line.split()
            score, image, sentence = float(line[0]), line[1], line[2:]
            image = image.split('/')
            filename = image[-1].rstrip(".jpg")
            segment = image[-2]
            video = image[-3]
            subset = image[-4] 
            uniq_hash = "/".join([subset, video, segment, filename])

            # skip scores below a certain threshold
            if score >= threshold:
                samples.append(uniq_hash)
                scores[uniq_hash] = score

        random.shuffle(samples) 
        a = int(len(samples)*splits[0]//100)
        b = a + int(len(samples)*splits[1]//100)
        train_set = samples[0:a]
        val_set = samples[a:b]
        test_set = samples[b:] 


        print("DEBUG a, b:", a, b)
        print("DEBUG train_set, val_set, test_set:", len(train_set), len(val_set), len(test_set))

        for sample in train_set:
            print("DEBUG:", sample)
            src = ""
            dst = ""
            subset,video,segment,filename = sample.split("/")
            npy_file = "_".join([video, segment, filename]) + ".npy"
            info_file = "_".join([video, segment, filename]) + "_info.npy"
            
            # resnet features
            if subset == "training":
                src = training_resnet_dir + "/" + npy_file
                dst = training_out_dir_resnet + "/" + npy_file
            elif subset == "validation":
                src = validation_resnet_dir + "/" + npy_file
                dst = training_out_dir_resnet + "/" + npy_file
            copyfile(src, dst)

            # faster rcnn features and info files
            if subset == "training":
                src = training_vmb_dir + "/" + npy_file
                dst = training_out_dir_vmb + "/" + npy_file
                copyfile(src, dst)
                src = training_vmb_dir + "/" + info_file
                dst = training_out_dir_vmb + "/" + info_file
                copyfile(src, dst)
            elif subset == "validation":
                src = validation_vmb_dir + "/" + npy_file
                dst = training_out_dir_vmb + "/" + npy_file
                copyfile(src, dst)
                src = validation_vmb_dir + "/" + info_file
                dst = training_out_dir_vmb + "/" + info_file
                copyfile(src, dst)

        for sample in val_set:
            print("DEBUG:", sample)
            src = ""
            dst = ""
            subset,video,segment,filename = sample.split("/")
            npy_file = "_".join([video, segment, filename]) + ".npy"
            info_file = "_".join([video, segment, filename]) + "_info.npy"
            
            # resnet features
            if subset == "training":
                src = training_resnet_dir + "/" + npy_file
                dst = validation_out_dir_resnet + "/" + npy_file
            elif subset == "validation":
                src = validation_resnet_dir + "/" + npy_file
                dst = validation_out_dir_resnet + "/" + npy_file
            copyfile(src, dst)

            # faster rcnn features and info files
            if subset == "training":
                src = training_vmb_dir + "/" + npy_file
                dst = validation_out_dir_vmb + "/" + npy_file
                copyfile(src, dst)
                src = training_vmb_dir + "/" + info_file
                dst = validation_out_dir_vmb + "/" + info_file
                copyfile(src, dst)
            elif subset == "validation":
                src = validation_vmb_dir + "/" + npy_file
                dst = validation_out_dir_vmb + "/" + npy_file
                copyfile(src, dst)
                src = validation_vmb_dir + "/" + info_file
                dst = validation_out_dir_vmb + "/" + info_file
                copyfile(src, dst)

        for sample in test_set:
            print("DEBUG:", sample)
            src = ""
            dst = ""
            subset,video,segment,filename = sample.split("/")
            npy_file = "_".join([video, segment, filename]) + ".npy"
            info_file = "_".join([video, segment, filename]) + "_info.npy"
            
            # resnet features
            if subset == "training":
                src = training_resnet_dir + "/" + npy_file
                dst = test_out_dir_resnet + "/" + npy_file
            elif subset == "validation":
                src = validation_resnet_dir + "/" + npy_file
                dst = test_out_dir_resnet + "/" + npy_file
            copyfile(src, dst)

            # faster rcnn features and info files
            if subset == "training":
                src = training_vmb_dir + "/" + npy_file
                dst = test_out_dir_vmb + "/" + npy_file
                copyfile(src, dst)
                src = training_vmb_dir + "/" + info_file
                dst = test_out_dir_vmb + "/" + info_file
                copyfile(src, dst)
            elif subset == "validation":
                src = validation_vmb_dir + "/" + npy_file
                dst = test_out_dir_vmb + "/" + npy_file
                copyfile(src, dst)
                src = validation_vmb_dir + "/" + info_file
                dst = test_out_dir_vmb + "/" + info_file
                copyfile(src, dst)
                   

if __name__ == "__main__":
    split_builder = SplitBuilder()
    split_builder.split_data()
