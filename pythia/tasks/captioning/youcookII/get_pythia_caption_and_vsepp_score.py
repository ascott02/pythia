import argparse
import glob
import json
import os
import numpy as np
import requests
import sys

# python get_pythia_caption_and_vsepp_score.py -o youcookII_pythia_json_files -i youcookII_json_files

from pythia.utils.text_utils import tokenize

# annotations_file = "/home/ascott/data/youcookII/YouCookII/annotations/youcookii_annotations_trainval.json"
# fcnn_features_dir = "/home/ascott/data/pythia/youcookII_features/training_features_vmb"
# resnet_features_dir = "/home/ascott/data/pythia/youcookII_features/training_features_resnet"

# input_file = "youcookII_json_files/youcookII_test_annotations.json"

pythia_url = "http://localhost:8080/api"
vsepp_url = "http://localhost:8081/api"
token = ""

class CaptionScoreBuilder:
    def __init__(self):
        self.args = self.get_args()

    def get_args(self):
        parser = argparse.ArgumentParser("Get Pythia caption and score for image in YouCookII")
        parser.add_argument(
            "-o",
            "--out_dir",
            type=str,
            default="./output_json_files",
            help="Output directory for JSON files",
        )
        parser.add_argument(
            "-i",
            "--input_dir",
            type=str,
            default="./input_json_files",
            help="JSON file with annotations for YouCookII",
        )

        return parser.parse_args()

    def get_rating(self, score):
        # b = (0.65507046 - .17) / 5
        # b
        # 0.097014092
        # .17 + b
        # 0.267014092
        # .17 + 2*b
        # 0.364028184
        # .17 + 3*b
        # 0.461042276
        # .17 + 4*b
        # 0.558056368
        rating = int()
        if score < 0.267014092:
            rating = 1
        elif score >= 0.267014092 and score < 0.364028184:
            rating = 2
        elif score >= 0.364028184 and score < 0.461042276:
            rating = 3
        elif score >= 0.461042276 and score < 0.558056368:
            rating = 4
        elif score >= 0.558056368:
            rating = 5
        return rating

    def get_caption(self, image_url):
        data = {"token": str(token), "image_url": str(image_url)}
        response = requests.post(pythia_url, data)
        return(response.text)

    def get_vsepp_score(self, image_url, caption):
        multipart_form_data = {
            'token': ('', str(token)),
            'caption': ('', str(caption)),
            'img_url': ('', str(image_url)),
            }
        response = requests.post(vsepp_url, files=multipart_form_data)
        return(response.text)

    def build(self):
        input_dir = self.args.input_dir
        out_dir = self.args.out_dir
        sentences = {}
        scores = {}
        url = "http://ec2-34-209-244-30.us-west-2.compute.amazonaws.com:8080/static/images/"
        
        for subset in ["training", "validation", "test"]:

            annotations = None
            annotations_file = input_dir + "/youcookII_" + subset + "_annotations.json"
            with open(annotations_file, "r") as f:
                annotations = json.load(f)

            if not os.path.isdir(out_dir):
                os.mkdir(out_dir)

            image_ids, filenames, coco_urls = [], [], []

            for entry in annotations["images"]:
                image_ids.append(entry['id'])
                filenames.append(entry['file_name'])
                coco_urls.append(entry['coco_url'])

            all_data = {}
            all_data["info"] = {"description": 'Default captions and vsepp scores for YouCookII ' + subset + ' set.'}
            all_data["images"] = []
            all_data["annotations"] = []
            for i, url in enumerate(coco_urls):
                images_dict = {}
                annotations_dict = {}

                images_dict['id'] = image_ids[i]
                images_dict['file_name'] = filenames[i]
                images_dict['coco_url'] = url

                caption = self.get_caption(url)
                if caption == '':
                    caption = '<unk>' 
                score = float(self.get_vsepp_score(url, caption))
                print("DEBUG id, url:", image_ids[i], url)
                print("DEBUG   score, caption:", score, caption)

                annotations_dict['image_id'] = image_ids[i]
                annotations_dict['caption'] = caption
                annotations_dict['vsepp_score'] = score
                annotations_dict['rating'] = self.get_rating(score)

                all_data["images"].append(images_dict)
                all_data["annotations"].append(annotations_dict)

            outfile = out_dir + "/youcookII_pythia_" + subset + "_annotations.json"
            with open(outfile, "w") as f:
                json.dump(all_data, f)



    '''
{
    "info": {
        "description": "GLAC captions with vsepp scores and scaled ratings"
    },
    "images": [
        {
            "id": 410328,
            "file_name": "COCO_val2014_000000410328.jpg",
            "coco_url": "http://images.cocodataset.org/val2014/COCO_val2014_000000410328.jpg"
        },

    "annotations": [
        {
            "image_id": 410328,
            "caption": "a baseball player is playing with a ball in the air .",
            "vsepp_score": "0.3354490622939134",
            "rating": "3"
        },
    '''

    def save_json(self, json_file):
        with open(self.args.out_file, "w") as f:
            json.dump(json_file, f)



if __name__ == "__main__":
    imdb_builder = CaptionScoreBuilder()
    imdb_builder.build()
