"""ComplexWebQuestions: A Dataset for Answering Complex Questions that Require Reasoning over Multiple Web Snippets."""

import json
import os

import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
    @inproceedings{Talmor2018TheWA,
            title={The Web as a Knowledge-Base for Answering Complex Questions},
            author={Alon Talmor and Jonathan Berant},
            booktitle={NAACL},
            year={2018}
        }
    """

_DESCRIPTION = """\
    ComplexWebQuestions is a dataset for answering complex questions that require reasoning over multiple web snippets. It contains a large set of complex questions in natural language, and can be used in multiple ways: 1) By interacting with a search engine, which is the focus of our paper (Talmor and Berant, 2018); 2) As a reading comprehension task: we release 12,725,989 web snippets that are relevant for the questions, and were collected during the development of our model; 3) As a semantic parsing task: each question is paired with a SPARQL query that can be executed against Freebase to retrieve the answer.
"""

_URL = "https://allenai.org/data/complexwebquestions"
_COMPLEXWEBQUESTIONS_URLS = {
    "train": "../DataPreprocessing4PromptKGQA/data/datasets/cwq/processed_train.json",
    "valid": "../DataPreprocessing4PromptKGQA/data/datasets/cwq/processed_valid.json",
    "test": "../DataPreprocessing4PromptKGQA/data/datasets/cwq/processed_test.json"
}

class ComplexWebQuestionsConfig(datasets.BuilderConfig):
    """BuilderConfig for ComplexWebQuestions"""
    def __init__(self,
                 data_url,
                 data_dir,
                 **kwargs):
        """BuilderConfig for ComplexWebQuestions.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(ComplexWebQuestionsConfig, self).__init__(**kwargs)
        self.data_url = data_url
        self.data_dir = data_dir

class ComplexWebQuestions(datasets.GeneratorBasedBuilder):
    """ComplexWebQuestions: A Dataset for Answering Complex Questions that Require Reasoning over Multiple Web Snippets."""
    BUILDER_CONFIGS = [
        ComplexWebQuestionsConfig(
            name="complex_web_questions",
            description="ComplexWebQuestions",
            data_url="",
            data_dir="ComplexWebQuestions"
        )
    ]

    def _info(self):
        features = datasets.Features(
                {
                    "uid": datasets.Value("string"),
                    "answers": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "query": datasets.Value("string"),
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

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            supervised_keys=None,
            homepage=_URL,
            citation=_CITATION,
            features=features
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_file": _COMPLEXWEBQUESTIONS_URLS["train"],
                    "split": "train"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_file": _COMPLEXWEBQUESTIONS_URLS["valid"],
                    "split": "validation"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_file": _COMPLEXWEBQUESTIONS_URLS["test"],
                    "split": "test"
                }
            )
        ]

    def _generate_examples(self, data_file, **kwargs):
        with open(data_file, encoding="utf8") as f:
            complexwebquestions = json.load(f)
            for idx, question in enumerate(complexwebquestions):
                question["dataset"] = "cwq"
                yield idx, question