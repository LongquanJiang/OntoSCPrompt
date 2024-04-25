"""WebQuestionsSP: The WebQuestions Semantic Parses Dataset"""

import json
import os

import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
    @inproceedings{yih-etal-2016-value,
        title={The Value of Semantic Parse Labeling for Knowledge Base Question Answering},
        author={Yih, Wen-tau and Richardson, Matthew and Meek, Chris and Chang, Ming-Wei and Suh, Jina},
        booktitle={Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)},
        year={2016},
        publisher={Association for Computational Linguistics},
        pages={201--206},
        }
    """

_DESCRIPTION = """\
    WebQuestionsSP Dataset Description
"""

_URL = "https://www.microsoft.com/en-us/download/details.aspx?id=52763"

_SIMPLE_DBPEDIA_QA_URLS = {
    "train": "../DataPreprocessing4PromptKGQA/data/datasets/simple_dbpedia_qa/processed_train.json",
    "valid": "../DataPreprocessing4PromptKGQA/data/datasets/simple_dbpedia_qa/processed_valid.json",
    "test": "../DataPreprocessing4PromptKGQA/data/datasets/simple_dbpedia_qa/processed_test.json"
}

class SimpleDBpediaQAConfig(datasets.BuilderConfig):
    """BuilderConfig for WebQuestionsSP"""
    def __init__(self,
                 data_url,
                 data_dir,
                 **kwargs):
        """BuilderConfig for WebQuestionsSP.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SimpleDBpediaQAConfig, self).__init__(**kwargs)
        self.data_url = data_url
        self.data_dir = data_dir

class SimpleDBpediaQA(datasets.GeneratorBasedBuilder):
    """WebQuestionsSP: The WebQuestions Semantic Parses Dataset"""
    BUILDER_CONFIGS = [
        SimpleDBpediaQAConfig(
            name="simple_dbpedia_qa",
            description="SimpleDBpediaQA",
            data_dir="SimpleDBpediaQA",
            data_url=""
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "uid": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "query": datasets.Value("string"),
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
            ),
            supervised_keys=None,
            homepage=_URL,
            citation=_CITATION
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_file": _SIMPLE_DBPEDIA_QA_URLS["train"],
                    "split": "train"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_file": _SIMPLE_DBPEDIA_QA_URLS["test"],
                    "split": "validation"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_file": _SIMPLE_DBPEDIA_QA_URLS["test"],
                    "split": "test"
                }
            )
        ]

    def _generate_examples(self, data_file, **kwargs):
        with open(data_file, encoding="utf8") as f:
            simple_dbpedia_qa = json.load(f)
            for idx, question in enumerate(simple_dbpedia_qa):
                question["dataset"] = "simple_dbpedia_qa"
                del question["PredicateList"]
                yield idx, question