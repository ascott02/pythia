# Copied from tasks/dialog/visdial/scripts/build_imdb.py
# and repurposed
import argparse
import glob
import json
import os
import numpy as np
import sys

# from pythia.utils.preprocessing import text_tokenize
from pythia.utils.text_utils import tokenize

# annotations_file = "/home/ascott/data/youcookII/YouCookII/annotations/youcookii_annotations_trainval.json"
# fcnn_features_dir = "/home/ascott/data/pythia/youcookII_features/training_features_vmb"
# resnet_features_dir = "/home/ascott/data/pythia/youcookII_features/training_features_resnet"


class IMDBBuilder:
    def __init__(self):
        self.args = self.get_args()

    def get_args(self):
        parser = argparse.ArgumentParser("Build IMDB for YouCookII")
        parser.add_argument(
            "-o",
            "--out_file",
            type=str,
            default="./imdb.npy",
            help="Output file for IMDB",
        )
        parser.add_argument(
            "-i",
            "--image_root",
            type=str,
            default="./COCO",
            help="Image directory for COCO",
        )
        """
        parser.add_argument(
            "-v", "--version", type=float, default=0.1, help="YouCookII Version"
        )
        """
        parser.add_argument(
            "-d",
            "--data_file",
            type=str,
            default="./annotations.json",
            help="JSON File with Annotations for YouCookII",
        )
        """
        parser.add_argument(
            "-s",
            "--set_type",
            type=str,
            default="train",
            help="Dataset type train|val|test",
        )
        """

        return parser.parse_args()

    def get_id_to_path_dict(self):
        id2path = {}
        globs = glob.iglob(os.path.join(self.args.image_root, "*", "*.npy"))
        # NOTE: based on assumption that image_id is unique across all splits
        for image_path in globs:
            path = "/".join(image_path.split("/")[-2:])
            image_id = int(image_path[-16:-4])
            id2path[image_id] = path

        return id2path

    def build(self):
        annotations_file = self.args.data_file
        image_dir = self.args.image_root
        
        # visdial_json_file = os.path.join(
        #     self.args.data_dir,
        #     "visdial_%.1f_%s.json" % (self.args.version, self.args.set_type),
        # )
        data = None

        with open(annotations_file, "r") as f:
            data = json.load(f)

        # final_questions = self.get_tokens(data["questions"])
        # final_answers = self.get_tokens(data["answers"])
        # dialogs = data["dialogs"]
        # dialogs_with_features = self.parse_dialogs(dialogs)

        # reference_tokens = []
        # caption_tokens = []
        # image_names = []
        # feature_paths = []
        # image_ids = []
        # caption_ids = []
        # caption_strs = []
               
        all_data = []
        # training_data = []
        # validation_data = []
        all_data.append({"metadata": 'youcookII'})
        # training_data.append({"metadata": 'youcookII', "subset": 'training'})
        # validation_data.append({"metadata": 'youcookII', "subset": 'validation'})
        counter = 0
        for video in data["database"]:
            for i in data["database"][video]["annotations"]:
                aDict = {}
                vid_seg = str(video) +  "_" + str(i['id'])
                feature_path = glob.glob(os.path.join(image_dir, vid_seg + "*.npy"))
                if len(feature_path) != 0:
                    feature_path = os.path.basename(feature_path[0])
                    print("DEBUG feature_path:", feature_path)
                    # sys.exit()
                    image_name = feature_path.rstrip(".npy")
                    image_id = counter
                    caption_id = counter
                    caption_str = i["sentence"]
                    # caption_token_list = []
                    caption_token_list = tokenize(caption_str)
                    caption_token_list.insert(0, "<s>")
                    caption_token_list.append("</s>")
          
                    # reference_tokens.append([caption_token_list])
                    # caption_tokens.append(caption_token_list)
                    # caption_strs.append(caption_str)
                    # caption_ids.append(caption_id)
                    # image_ids.append(image_id)
                    # image_names.append(image_name)
                    # feature_paths.append(feature_path)

                    aDict["reference_tokens"] = [caption_token_list]
                    aDict["caption_tokens"] = caption_token_list
                    aDict["caption_str"] = caption_str
                    aDict["caption_id"] = caption_id
                    aDict["image_id"] = image_id
                    aDict["image_name"] = image_name
                    aDict["feature_path"] = feature_path

                    # print("DEBUG subset:", data["database"][video]["subset"])
                    # sys.exit()
            
                    # if str(data["database"][video]["subset"]) == "training":
                    #     training_data.append(aDict)
                    # elif str(data["database"][video]["subset"]) == "validation":
                    #     validation_data.append(aDict)

                    all_data.append(aDict)
                    counter+=1
                    
        """
        imdb = {
            # "questions": final_questions,
            # "answers": final_answers,
            # "dialogs": dialogs_with_features,
            "reference_tokens": reference_tokens,
            "caption_tokens": caption_tokens,
            "image_name": image_names,
            "feature_path": feature_paths,
            "image_id": image_ids,
            "caption_id": caption_ids,
            "caption_str": caption_strs,
        }

        np_data = np.array(list(zip( \
            list(zip(["reference_tokens"]*len(reference_tokens), reference_tokens)),
            list(zip(["caption_tokens"]*len(caption_tokens), caption_tokens)),
            list(zip(["image_name"]*len(image_names), image_names)),
            list(zip(["feature_path"]*len(feature_paths), feature_paths)),
            list(zip(["image_id"]*len(image_ids), image_ids)),
            list(zip(["caption_id"]*len(caption_ids), caption_ids)),
            list(zip(["caption_str"]*len(caption_strs), caption_strs)),
            )))
        """

        np.save(self.args.out_file, np.array(all_data))
        # np.save("imdb_youcookII_training.npy", np.array(training_data))
        # np.save("imdb_youcookII_validation.npy", np.array(validation_data))

        # np.save(self.args.out_file, np_data)
        # self.save_imdb(imdb)

    def save_imdb(self, imdb):
        with open(self.args.out_file, "w") as f:
            json.dump(imdb, f)

    def get_tokens(self, sentences):
        if not isinstance(sentences, list):
            sentences = [sentences]
        final_sentences = []
        for idx, sentence in enumerate(sentences):
            tokens = text_tokenize(sentence)
            final_sentences.append(tokens)

        return final_sentences

    def parse_dialogs(self, dialogs):
        id2path = self.get_id_to_path_dict()

        for dialog in dialogs:
            image_id = dialog["image_id"]
            image_feature_path = id2path[image_id]
            dialog["image_feature_path"] = image_feature_path
            dialog["caption"] = self.get_tokens(dialog["caption"])

        return dialogs


if __name__ == "__main__":
    imdb_builder = IMDBBuilder()
    imdb_builder.build()
