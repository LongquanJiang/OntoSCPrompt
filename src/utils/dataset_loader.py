
from typing import Tuple
import logging
import json
import datasets
from datasets.dataset_dict import DatasetDict
from datasets.metric import Metric
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.training_args import TrainingArguments
from .args import *
from .dataset import (
    DatasetSplits,
    prepare_splits
)

from .webqsp import webqsp_add_serialized_schema, webqsp_pre_process_function
from .dblp_qald import dblp_qald_add_serialized_schema, dblp_qald_pre_process_function
from .cwq import cwq_add_serialized_schema, cwq_pre_process_function
from .lcquad import lcquad_add_serialized_schema, lcquad_pre_process_function
from .simple_dbpedia_qa import simple_dbpedia_qa_add_serialized_schema, simple_dbpedia_qa_pre_process_function


logger = logging.getLogger(__name__)


def load_dataset(
        data_args: DataArguments,
        model_args: ModelArguments,
        data_training_args: DataTrainingArguments,
        training_args: TrainingArguments,
        tokenizer: PreTrainedTokenizerFast,
) -> Tuple[Metric, DatasetSplits]:

    ontologies = dict()

    if data_training_args.level_ontology == "dataset":
        ontology_paths = data_training_args.dataset_ontology_paths
    elif data_training_args.level_ontology == "kb":
        ontology_paths = data_training_args.kb_ontology_paths

    for id, path in ontology_paths.items():
        ontologies.update({id: json.load(open(path, "r"))})

    ########LC-QuAD#########
    _lcquad_dataset_dict: Callable[[], DatasetDict] = lambda: datasets.load.load_dataset(
        path=data_args.dataset_paths["lcquad"], cache_dir=model_args.cache_dir
    )
    _lcquad_metric: Callable[[], Metric] = lambda: datasets.load.load_metric(
        "exact_match", config_name=data_args.metric_config
    )
    _lcquad_add_serialized_schema = lambda ex, mode: lcquad_add_serialized_schema(
        ex=ex,
        mode=mode,
        ontologies=ontologies,
        data_training_args=data_training_args,
    )
    _lcquad_pre_process_function = lambda batch, mode: lcquad_pre_process_function(
        batch=batch,
        mode=mode,
        data_training_args=data_training_args,
        tokenizer=tokenizer,
    )

    ########WebQuestionSP#########
    _webqsp_dataset_dict: Callable[[], DatasetDict] = lambda: datasets.load.load_dataset(
        path=data_args.dataset_paths["webqsp"], cache_dir=model_args.cache_dir
    )
    _webqsp_metric: Callable[[], Metric] = lambda: datasets.load.load_metric(
        "exact_match", config_name=data_args.metric_config
    )
    _webqsp_add_serialized_schema = lambda ex, mode: webqsp_add_serialized_schema(
        ex=ex,
        mode=mode,
        ontologies=ontologies,
        data_training_args=data_training_args,
    )
    _webqsp_pre_process_function = lambda batch, mode: webqsp_pre_process_function(
        batch=batch,
        mode=mode,
        data_training_args=data_training_args,
        tokenizer=tokenizer
    )

    ########ComplexWebQuestions#########
    _cwq_dataset_dict: Callable[[], DatasetDict] = lambda: datasets.load.load_dataset(
        path=data_args.dataset_paths["cwq"], cache_dir=model_args.cache_dir
    )
    _cwq_metric: Callable[[], Metric] = lambda: datasets.load.load_metric(
        "exact_match", config_name=data_args.metric_config
    )
    _cwq_add_serialized_schema = lambda ex, mode: cwq_add_serialized_schema(
        ex=ex,
        mode=mode,
        ontologies=ontologies,
        data_training_args=data_training_args,
    )
    _cwq_pre_process_function = lambda batch, mode: cwq_pre_process_function(
        batch=batch,
        mode=mode,
        data_training_args=data_training_args,
        tokenizer=tokenizer,
    )

    ########DBLP-QALD#########
    _dblp_qald_dataset_dict: Callable[[], DatasetDict] = lambda: datasets.load.load_dataset(
        data_args.dataset_paths["dblp_qald"], cache_dir=model_args.cache_dir
    )
    _dblp_qald_metric: Callable[[], Metric] = lambda: datasets.load.load_metric(
        "exact_match", config_name=data_args.metric_config
    )
    _dblp_qald_add_serialized_schema = lambda ex, mode: dblp_qald_add_serialized_schema(
        ex=ex,
        mode=mode,
        ontologies=ontologies,
        data_training_args=data_training_args,
    )
    _dblp_qald_pre_process_function = lambda batch, mode: dblp_qald_pre_process_function(
        batch=batch,
        mode=mode,
        data_training_args=data_training_args,
        tokenizer=tokenizer,
    )

    ########Simple DBpedia QA#########
    _simple_dbpedia_qa_dataset_dict: Callable[[], DatasetDict] = lambda: datasets.load.load_dataset(
        data_args.dataset_paths["simple_dbpedia_qa"], cache_dir=model_args.cache_dir
    )
    _simple_dbpedia_qa_metric: Callable[[], Metric] = lambda: datasets.load.load_metric(
        "exact_match", config_name=data_args.metric_config
    )
    _simple_dbpedia_qa_add_serialized_schema = lambda ex, mode: simple_dbpedia_qa_add_serialized_schema(
        ex=ex,
        mode=mode,
        ontologies=ontologies,
        data_training_args=data_training_args,
    )
    _simple_dbpedia_qa_pre_process_function = lambda batch, mode: simple_dbpedia_qa_pre_process_function(
        batch=batch,
        mode=mode,
        data_training_args=data_training_args,
        tokenizer=tokenizer,
    )


    _prepare_splits_kwargs = {
        "data_args": data_args,
        "training_args": training_args,
        "data_training_args": data_training_args,
    }

    if data_args.dataset == "lcquad2":
        metric = _lcquad2_metric()
        dataset_splits = prepare_splits(
            dataset_dict=_lcquad2_dataset_dict(),
            add_serialized_schema=_lcquad2_add_serialized_schema,
            pre_process_function=_lcquad2_pre_process_function,
            **_prepare_splits_kwargs,
        )
    elif data_args.dataset == "webqsp":
        metric = _webqsp_metric()
        dataset_splits = prepare_splits(
            dataset_dict=_webqsp_dataset_dict(),
            add_serialized_schema=_webqsp_add_serialized_schema,
            pre_process_function=_webqsp_pre_process_function,
            **_prepare_splits_kwargs,
        )
    elif data_args.dataset == "dblp_qald":
        metric = _dblp_qald_metric()
        dataset_splits = prepare_splits(
            dataset_dict=_dblp_qald_dataset_dict(),
            add_serialized_schema=_dblp_qald_add_serialized_schema,
            pre_process_function=_dblp_qald_pre_process_function,
            **_prepare_splits_kwargs,
        )
    elif data_args.dataset == "cwq":
        metric = _cwq_metric()
        dataset_splits = prepare_splits(
            dataset_dict=_cwq_dataset_dict(),
            add_serialized_schema=_cwq_add_serialized_schema,
            pre_process_function=_cwq_pre_process_function,
            **_prepare_splits_kwargs,
        )
    elif data_args.dataset == "graphquestions":
        metric = _graphquestions_metric()
        dataset_splits = prepare_splits(
            dataset_dict=_graphquestions_dataset_dict(),
            add_serialized_schema=_graphquestions_add_serialized_schema,
            pre_process_function=_graphquestions_pre_process_function,
            **_prepare_splits_kwargs,
        )
    elif data_args.dataset == "lcquad":
        metric = _lcquad_metric()
        dataset_splits = prepare_splits(
            dataset_dict=_lcquad_dataset_dict(),
            add_serialized_schema=_lcquad_add_serialized_schema,
            pre_process_function=_lcquad_pre_process_function,
            **_prepare_splits_kwargs,
        )
    elif data_args.dataset == "simple_dbpedia_qa":
        metric = _simple_dbpedia_qa_metric()
        dataset_splits = prepare_splits(
            dataset_dict=_simple_dbpedia_qa_dataset_dict(),
            add_serialized_schema=_simple_dbpedia_qa_add_serialized_schema,
            pre_process_function=_simple_dbpedia_qa_pre_process_function,
            **_prepare_splits_kwargs,
        )
    else:
        raise NotImplementedError()

    return metric, dataset_splits
