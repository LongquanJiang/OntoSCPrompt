"""LC-QuAD: A Large Scale Complex Question Answering Dataset."""

import json
import os

import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
    @inproceedings{trivedi2017lc,
        title={Lc-quad: A corpus for complex question answering over knowledge graphs},
        author={Trivedi, Priyansh and Maheshwari, Gaurav and Dubey, Mohnish and Lehmann, Jens},
        booktitle={International Semantic Web Conference},
        pages={210--218},
        year={2017},
        organization={Springer}
    }
    """

_DESCRIPTION = """\
    LC-QuAD is a Question Answering dataset with 5000 pairs of question and its corresponding SPARQL query. The target knowledge base is DBpedia, specifically, the April, 2016 version. Please see our paper for details about the dataset creation process and framework.
"""

_URL = "http://lc-quad.sda.tech/lcquad1.0.html"
_LCQUAD_URLS = {
    "train": "../DataPreprocessing4PromptKGQA/data/datasets/lcquad/processed_train.json",
    "test": "../DataPreprocessing4PromptKGQA/data/datasets/lcquad/processed_test.json"
}

class LCQuADConfig(datasets.BuilderConfig):
    """BuilderConfig for LC-QuAD"""
    def __init__(self,
                 data_url,
                 data_dir,
                 **kwargs):
        """BuilderConfig for LC-QuAD.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(LCQuADConfig, self).__init__(**kwargs)
        self.data_url = data_url
        self.data_dir = data_dir

class LCQuAD(datasets.GeneratorBasedBuilder):
    """LC-QuAD: A Large Scale Complex Question Answering Dataset."""
    BUILDER_CONFIGS = [
        LCQuADConfig(
            name="lcquad",
            description="LCQuAD",
            data_url="",
            data_dir="LCQuAD"
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            supervised_keys=None,
            homepage=_URL,
            citation=_CITATION,
            features=datasets.Features(
                {
                    "uid": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "query": datasets.Value("string"),
                    "answers": datasets.Value("string"),
                    "original_query": datasets.Value("string"),
                    "entities": datasets.Sequence(
                        datasets.Value("string")
                    ),
                    "relations": datasets.Sequence(
                        datasets.Value("string")
                    ),
                    "structure": datasets.Value("string"),
                    "content": datasets.Value("string"),
                    "kb": datasets.Value("string"),
                    "dataset": datasets.Value("string")
                }
            )
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_file": _LCQUAD_URLS["train"],
                    "split": "train"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_file": _LCQUAD_URLS["test"],
                    "split": "validation"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_file": _LCQUAD_URLS["test"],
                    "split": "test"
                }
            )
        ]

    def _generate_examples(self, data_file, **kwargs):
        with open(data_file, encoding="utf8") as f:
            all_data = json.load(f)
            for idx, example in enumerate(all_data):
                example["dataset"] = "lcquad"
                yield idx,example