# copied from visdial example
import json

from pythia.scripts.extract_vocabulary import ExtractVocabulary

"""
{
    "database": {
        "--bv0V6ZjWI": {
            "annotations": [
                {
                    "id": 0,
                    "segment": [
                        8,
                        23
                    ],
                    "sentence": "crush and chop the garlic"
"""


class ExtractYouCookIIVocabulary(ExtractVocabulary):
    def __init__(self):
        super(ExtractYouCookIIVocabulary, self).__init__()

    def get_text(self):
        text = []

        for input_file in self.input_files:
            with open(input_file, "r") as f:
                f_json = json.load(f)
                # Add 'captions' from youcookII
                for video in f_json["database"]:
                    for annotation in f_json["database"][video]["annotations"]:
                
                        print("DEBUG annoation['sentence']:", annotation["sentence"])
                        text += [annotation["sentence"]]

        return text


if __name__ == "__main__":
    extractor = ExtractYouCookIIVocabulary()
    extractor.extract()
