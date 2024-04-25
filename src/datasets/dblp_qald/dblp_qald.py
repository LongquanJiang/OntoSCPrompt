# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""DBLP-QuAD: A Question Answering Dataset over the DBLP Scholarly Knowledge Graph."""


import json
import os

import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = """
    @article{DBLP-QuAD,
        title={DBLP-QuAD: A Question Answering Dataset over the DBLP Scholarly Knowledge Graph},
        author={Banerjee, Debayan and Awale, Sushil and Usbeck, Ricardo and Biemann, Chris},
        year={2023}
"""

_DESCRIPTION = """\
    DBLP-QuAD is a scholarly knowledge graph question answering dataset with \
    10,000 question - SPARQL query pairs targeting the DBLP knowledge graph. \
    The dataset is split into 7,000 training, 1,000 validation and 2,000 test \
    questions.
"""

_URL = "https://zenodo.org/record/7643971/files/DBLP-QuAD.zip"

_DBLP_QALD_FILES = {
    "train": "../DataPreprocessing4PromptKGQA/data/datasets/dblp_qald/processed_train.json",
    "valid": "../DataPreprocessing4PromptKGQA/data/datasets/dblp_qald/processed_valid.json",
    "test": "../DataPreprocessing4PromptKGQA/data/datasets/dblp_qald/processed_test.json"
}

class DBLPQuAD(datasets.GeneratorBasedBuilder):
    """
        DBLP-QuAD: A Question Answering Dataset over the DBLP Scholarly Knowledge Graph.
    """

    VERSION = datasets.Version("1.1.0")

    def _info(self):
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datasets.Features(
                {
                    "uid": datasets.Value("string"),
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
            ),
            supervised_keys=None,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": _DBLP_QALD_FILES["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": _DBLP_QALD_FILES["valid"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": _DBLP_QALD_FILES["test"]},
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""

        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for id_, row in enumerate(data):
                row["dataset"] = "dblp_qald"
                yield id_, row