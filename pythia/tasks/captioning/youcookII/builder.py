from pythia.common.registry import registry
from pythia.tasks.vqa.vqa2 import VQA2Builder

from .dataset import YouCookIIDataset

@registry.register_builder("youcookII")
class YouCookIIBuilder(VQA2Builder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "youcookII"
        self.set_dataset_class(YouCookIIDataset)

    def update_registry_for_model(self, config):
        registry.register(
            self.dataset_name + "_text_vocab_size",
            self.dataset.text_processor.get_vocab_size(),
        )
