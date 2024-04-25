# Set up logging
import sys
import logging
import wandb

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.WARNING,
)
logger = logging.getLogger(__name__)

import os
import json
from pathlib import Path
from contextlib import nullcontext
from dataclasses import asdict
from transformers.hf_argparser import HfArgumentParser
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from transformers.models.auto import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers.trainer_utils import get_last_checkpoint, set_seed
from transformers.models.longt5.modeling_longt5 import LongT5ForConditionalGeneration
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from tokenizers import AddedToken
from utils.args import ModelArguments, DataTrainingArguments, DataArguments, WandDBArguments
from utils.decode_wrapper import with_grammar
from utils.dataset_loader import load_dataset
from utils.webqsp import WebQSPTrainer
from utils.dblp_qald import DblpQALDTrainer
from utils.cwq import CWQTrainer
from utils.lcquad import LCQuADTrainer
from utils.simple_dbpedia_qa import SimpleDBpediaQATrainer
from utils.PT_wrapper import PromptWrapper

from utils.transformers_cfg.grammar_utils import VanillaGrammarConstraint

set_seed(1314)

# set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"] = "seq2sparql-prompt"
# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"] = "false"
# turn off watch to log faster
os.environ["WANDB_WATCH"] = "false"

def main() -> None:
    # See all possible arguments by passing the --help flag to this script.
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, DataTrainingArguments, Seq2SeqTrainingArguments, WandDBArguments)
    )
    model_args: ModelArguments
    data_args: DataArguments
    data_training_args: DataTrainingArguments
    training_args: Seq2SeqTrainingArguments
    wandb_args: WandDBArguments
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, data_training_args, training_args, wandb_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    elif len(sys.argv) == 3 and sys.argv[1].startswith("--local_rank") and sys.argv[2].endswith(".json"):
        data = json.loads(Path(os.path.abspath(sys.argv[2])).read_text())
        data.update({"local_rank": int(sys.argv[1].split("=")[1])})
        model_args, data_args, data_training_args, training_args, wandb_args = parser.parse_dict(args=data)
    else:
        model_args, data_args, data_training_args, training_args, wandb_args = parser.parse_args_into_dataclasses()

    combined_args_dict = {
        **asdict(model_args),
        **asdict(data_args),
        **asdict(data_training_args),
        **training_args.to_sanitized_dict(),
    }
    combined_args_dict.pop("local_rank", None)

    # Detect last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # set path
    fewshot_identifier = f'R{str(int(data_training_args.train_samples_ratio * 100))}'
    training_args.output_dir = os.path.join(training_args.output_dir, f'{fewshot_identifier}_results')
    if training_args.do_train:
        if data_training_args.structure_path == "":
            data_training_args.structure_path = os.path.join(training_args.output_dir, f'structure/structure.json')
        if data_training_args.training_method == "PFT" and data_training_args.initial_vectors_path == "":
            data_training_args.initial_vectors_path = os.path.join(training_args.output_dir,
                                                                   f'{data_training_args.stage}/head.npy')
        if data_training_args.use_decomposition:
            training_args.output_dir = os.path.join(training_args.output_dir, data_training_args.stage)
        else:
            training_args.output_dir = os.path.join(training_args.output_dir, 'seq2seq')
    elif training_args.do_eval or training_args.do_predict:
        model_args.model_name_or_path = os.path.join(training_args.output_dir, f'{data_training_args.stage}/BEST_MODEL')
        training_args.output_dir = os.path.join(training_args.output_dir, 'prediction')
        if data_training_args.stage == "content" and os.path.exists(
                os.path.join(training_args.output_dir, "hypotheses.json")):
            os.remove(os.path.join(training_args.output_dir, "hypotheses.json"))
        if data_training_args.structure_path == "":
            data_training_args.structure_path = os.path.join(training_args.output_dir, f'structure.json')
        if data_training_args.training_method == "PFT" and data_training_args.initial_vectors_path == "":
            data_training_args.initial_vectors_path = os.path.join(model_args.model_name_or_path, 'head.npy')

    os.makedirs(training_args.output_dir, exist_ok=True)


    if training_args.do_train:
        # save your trained model checkpoint to wandb
        os.environ["WANDB_LOG_MODEL"] = "true"
        ## Initialize Wandb
        run_name = f'{model_args.model_name_or_path}-{data_args.dataset}-{data_training_args.stage}-{data_training_args.training_method}'
        wandb.init(
            name=run_name,
            group=data_training_args.experiment_name,
        )
        training_args.report_to = ['wandb']


    if training_args.local_rank <= 0:
        with open(f"{training_args.output_dir}/combined_args.json", "w") as f:
            json.dump(combined_args_dict, f, indent=4)

    # Initialize config
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        max_length=data_training_args.max_target_length,
        num_beams=data_training_args.num_beams,
        num_beam_groups=data_training_args.num_beam_groups,
        diversity_penalty=data_training_args.diversity_penalty,
        gradient_checkpointing=training_args.gradient_checkpointing,
        use_cache=not training_args.gradient_checkpointing,
        num_return_sequences=data_training_args.num_beams if data_training_args.use_constrained_decoding and data_training_args.stage == "content" else 1,
    )

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    assert isinstance(tokenizer, PreTrainedTokenizerFast), "Only fast tokenizers are currently supported"
    if isinstance(tokenizer, T5TokenizerFast):
        # In T5 `<` is OOV, see https://github.com/google-research/language/blob/master/language/nqg/tasks/spider/restore_oov.py
        tokenizer.add_tokens([
            AddedToken(" <="),
            AddedToken(" <")
        ])

    print("Load dataset")
    metric, dataset_splits = load_dataset(
        data_args=data_args,
        model_args=model_args,
        data_training_args=data_training_args,
        training_args=training_args,
        tokenizer=tokenizer,
    )
    print("Load dataset")

    if training_args.do_train:
        if data_training_args.training_method == 'PT':
            training_args.eval_steps = 100 * int(dataset_splits.train_split.dataset.num_rows / (
                        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps))
        else:
            training_args.eval_steps = 2 * int(dataset_splits.train_split.dataset.num_rows / (
                        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps))
        training_args.save_steps = training_args.eval_steps * 100000

    # Load grammar
    with open("../DataPreprocessing4PromptKGQA/data/grammar/sparql_structure.ebnf", "r") as file:
        grammar_str = file.read()
    grammar_constraint = VanillaGrammarConstraint(grammar_str, "start", tokenizer)

    if data_training_args.use_constrained_decoding:
        model_cls_wrapper = lambda model_cls: with_grammar(
            model_cls=model_cls, tokenizer=tokenizer,
            grammar_constraint=grammar_constraint, stage=data_training_args.stage,
            subgraphs=dataset_splits.subgraphs
        )
    else:
        model_cls_wrapper = lambda model_cls: model_cls

    model_ = model_cls_wrapper(AutoModelForSeq2SeqLM).from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if isinstance(model_, LongT5ForConditionalGeneration):
        model_.resize_token_embeddings(len(tokenizer))
    if data_training_args.stage == 'structure':
        if data_args.dataset in ["lcquad", "webqsp", "simple_dbpedia_qa", "cwq"]:
            prompt_length_list = [60, 15, 15, 60]
        else:
            prompt_length_list = [40, 10, 10, 40]
    elif data_training_args.stage == 'content':
        prompt_length_list = [60, 40, 40, 60]
    if data_training_args.training_method == 'PT':
        print(f"-------PT--------")
        model = PromptWrapper(
            model_,
            use_constrained_decoding=data_training_args.use_constrained_decoding,
            prompt_length_list=prompt_length_list,
            freeze_model=True,
            initialize_from_vocab=True,
        )
        model.main_input_name = 'input_ids'
    elif data_training_args.training_method == 'PFT':
        print(f"-------PFT--------")
        model = PromptWrapper(
            model_,
            use_constrained_decoding=data_training_args.use_constrained_decoding,
            prompt_length_list=prompt_length_list,
            freeze_model=False,
            stage=data_training_args.stage,
            initial_vectors_path=data_training_args.initial_vectors_path,
            initialize_from_pretrain=True,
        )
        model.main_input_name = 'input_ids'
    else:
        print("-------FT--------")
        model = model_

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    print("Initialize Trainer")
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "use_decomposition": data_training_args.use_decomposition,
        "training_method": data_training_args.training_method,
        "stage": data_training_args.stage,
        "metric": metric,
        "train_dataset": dataset_splits.train_split.dataset if training_args.do_train else None,
        "eval_dataset": dataset_splits.eval_split.dataset if training_args.do_eval else None,
        "eval_examples": dataset_splits.eval_split.examples if training_args.do_eval else None,
        "tokenizer": tokenizer,
        "data_collator": DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=(-100 if data_training_args.ignore_pad_token_for_loss else tokenizer.pad_token_id),
            pad_to_multiple_of=8 if training_args.fp16 else None,
        ),
        "ignore_pad_token_for_loss": data_training_args.ignore_pad_token_for_loss
    }

    if data_args.dataset in ["webqsp"]:
        trainer = WebQSPTrainer(**trainer_kwargs)
    elif data_args.dataset in ["dblp_qald"]:
        trainer = DblpQALDTrainer(**trainer_kwargs)
    elif data_args.dataset in ["cwq"]:
        trainer = CWQTrainer(**trainer_kwargs)
    elif data_args.dataset in ["lcquad"]:
        trainer = LCQuADTrainer(**trainer_kwargs)
    elif data_args.dataset in ["simple_dbpedia_qa"]:
        trainer = SimpleDBpediaQATrainer(**trainer_kwargs)
    else:
        raise NotImplementedError()

    # Training
    if training_args.do_train:
        logger.info("*** Train ***")

        checkpoint = None

        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(
            max_length=data_training_args.val_max_target_length,
            max_time=data_training_args.val_max_time,
            num_beams=data_training_args.num_beams,
            metric_key_prefix="eval",
        )
        metrics["eval_samples"] = dataset_splits.eval_split.dataset.num_rows

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Testing
    if training_args.do_predict:
        logger.info("*** Predict ***")
        for section, test_split in dataset_splits.test_splits.items():
            results = trainer.predict(
                test_split.dataset,
                test_split.examples,
                max_length=data_training_args.val_max_target_length,
                max_time=data_training_args.val_max_time,
                num_beams=data_training_args.num_beams,
                metric_key_prefix=section)
            metrics = results.metrics

            metrics[f"{section}_samples"] = len(test_split.dataset)

            trainer.log_metrics(section, metrics)
            trainer.save_metrics(section, metrics)

    if training_args.do_train:
        wandb.finish()


if __name__ == "__main__":
    main()
