
from typing import Optional, List, Dict, Callable
from dataclasses import dataclass, field

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."}
    )
    use_auth_token: bool = field(
        default=False,
        metadata={"help": "Will use the token generated when running `transformers-cli login` (necessary to use this script with private models)."}
    )



@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."}
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={"help": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."}
    )
    max_target_length: Optional[int] = field(
        default=512,
        metadata={"help": "The maximum total sequence length for target text after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."}
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum total sequence length for validation target text after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`. This argument is also used to override the `max_length` param of `model.generate`, which is used during `evaluate` and `predict`."}
    )
    val_max_time: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum allowed time in seconds for generation of one example. This setting can be used to stop generation whenever the full generation exceeds the specified amount of time."}
    )
    train_samples_ratio: Optional[float] = field(
        default=1.,
        metadata={"help": "For few-shot learning value if set."}
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate the number of validation or test examples to this value if set."}
    )
    use_constrained_decoding: bool = field(
        default=True,
        metadata={"help": "Whether constrained decoding is used. which is used during ``structure-stage`` and ``content-stage``."}
    )
    num_beams: int = field(
        default=1,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )
    num_return_sequences: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of return_sequences to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )
    num_beam_groups: int = field(
        default=1,
        metadata={
            "help": "Number of beam groups to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )
    do_sample: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether"
        }
    )
    diversity_penalty: Optional[float] = field(
        default=None,
        metadata={
            "help": "Diversity penalty to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )
    repetition_penalty: Optional[float] = field(
        default=None,
        metadata={
            "help": "Repetition penalty to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None,
        metadata={"help": "A prefix to add before every source text (useful for T5 models)."},
    )
    schema_serialization_type: str = field(
        default="peteshaw",
        metadata={"help": "Choose between ``verbose`` and ``peteshaw`` schema serialization."},
    )
    schema_serialization_with_db_id: bool = field(
        default=True,
        metadata={"help": "Whether or not to add the database id to the context. Needed for Picard."},
    )
    schema_serialization_with_prompt: str = field(
        default="",
        metadata={"help": "Whether or not to use prompt."}
    )
    schema_serialization_with_db_content: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the database content to resolve field matches."},
    )
    schema_serialization_with_db_concept_content: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the database property content to resolve field matches."},
    )
    schema_serialization_with_db_relation_content: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the database property content to resolve field matches."},
    )
    schema_serialization_with_db_entity_content: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the database entity content to resolve field matches."},
    )
    normalize_query: bool = field(
        default=True,
        metadata={"help": "Whether to normalize the SPARQL queries."}
    )
    target_with_db_id: bool = field(
        default=True,
        metadata={"help": "Whether or not to add the database id to the target. Needed for Picard."},
    )
    use_decomposition: bool = field(
        default=True,
        metadata={"help": "Whether to use decomposition."}
    )
    stage: str = field(
        default='structure',
        metadata={"help": "Training structure prediction module or content prediction module."}
    )
    training_method: str = field(
        default='FT',
        metadata={"help": "Training with PT or FT."}
    )

    grammar_path: str = field(
        default="",
        metadata={"help": ""}
    )
    structure_path: str = field(
        default='',
        metadata={"help": "The path to the sparql structure. Only use in the content-fill stage."}
    )
    initial_vectors_path: str = field(
        default="",
        metadata={"help": "the path to the initial learnable vectors. only use in the 'PFT' mode."}
    )
    level_ontology: str = field(
        default="dataset",
        metadata={"help": "The level of ontology to used. dataset or kb"}
    )
    dataset_ontology_paths: Dict[str, str] = field(
        default_factory=lambda: {
            "webqsp": "../DataPreprocessing4PromptKGQA/data/datasets/webqsp/ontology_webqsp.json",
            "cwq": "../DataPreprocessing4PromptKGQA/data/datasets/cwq/ontology_cwq_webqsp.json",
            "lcquad": "../DataPreprocessing4PromptKGQA/data/datasets/lcquad/ontology_lcquad.json",
            "simple_dbpedia_qa": "../DataPreprocessing4PromptKGQA/data/datasets/simple_dbpedia_qa/ontology_simple_dbpedia_qa.json",
            "dblp_qald": "../DataPreprocessing4PromptKGQA/data/datasets/dblp_qald/ontology_dblp_qald.json"
        },
        metadata={"help": "Paths of the dataset modules."}
    )
    kb_ontology_paths: Dict[str, str] = field(
        default_factory=lambda: {
            "coypukg": "../DataPreprocessing4PromptKGQA/data/graphs/coypukg/ontology_coypukg.json",
            "freebase": "../DataPreprocessing4PromptKGQA/data/graphs/freebase/ontology_freebase.json",
            "dbpedia": "../DataPreprocessing4PromptKGQA/data/graphs/dbpedia/ontology_dbpedia.json"
        },
        metadata={"help": "Paths of the dataset modules."}
    )
    structure_granularity: str = field(
        default="fine",
        metadata={"help": "Which level of granuality should be used for SPARQL structure. "}
    )
    experiment_name: str = field(
        default="experiment_1",
        metadata={"help": "Experiment name"}
    )
    def __post_init__(self):
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


@dataclass
class DataArguments:
    dataset: str = field(
        metadata={"help": "The dataset to be used."}
    )
    dataset_paths: Dict[str, str] = field(
        default_factory=lambda: {
            "webqsp": "./src/datasets/webqsp",
            "cwq": "./src/datasets/cwq",
            "lcquad": "./src/datasets/lcquad",
            "dblp_qald": "./src/datasets/dblp_qald",
            "simple_dbpedia_qa": "./src/datasets/simple_dbpedia_qa"
        },
        metadata={"help": "Paths of the dataset modules."}
    )
    metric_config: str = field(
        default="exact_match",
        metadata={"help": "Choose between `èxact_match``"}
    )
    metric_paths: Dict[str, str] = field(
        default_factory=lambda: {
            "webqsp": "./src/metrics/webqsp",
            "cwq": "./src/metrics/cwq",
            "dblp_qald": "./src/metrics/dblp_qald",
            "lcquad": "./src/metrics/lcquad"
        },
        metadata={"help": "Paths of the metric modules."}
    )
    data_config_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to data configuration file (specifying the database splits)"}
    )

@dataclass
class WandDBArguments():
    use_wandb: bool = field(
        default=True,
        metadata={"help": "indicates whether to use wandb."}
    )