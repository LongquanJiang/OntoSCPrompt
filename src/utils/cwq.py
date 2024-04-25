
import os
import json
import numpy as np
from .trainer import Seq2SeqTrainer, EvalPrediction
from datasets.arrow_dataset import Dataset
from .dataset import serialize_schema, encode, decode, combine_SC
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from .args import *

def cwq_get_input(
        question: str,
        serialized_schema: str,
        prefix: str,
) -> str:
    return prefix + question.strip() + " | " + serialized_schema.strip()


def cwq_get_target(
        query: str,
        db_id: str,
        target_with_db_id: bool,
) -> str:
    return f"{db_id} | {query}" if target_with_db_id else query


def cwq_add_serialized_schema(
        ex: dict,
        mode: Optional[str],
        ontologies: dict,
        data_training_args: DataTrainingArguments
) -> dict:
    ontology = ontologies["cwq"]

    ent2label = {rel["qid"].split(":")[1]: rel["label"] for rel in ontology["entities"]}
    rel2label = {rel["qid"]: rel["label"] for rel in ontology["relations"]}
    concept2label = {concept["qid"]: concept["label"] for concept in ontology["concepts"]}

    serialized_schema = serialize_schema(
        db_id=ex["kb"],
        db_concepts=[qid for qid in concept2label.keys()],
        db_concept_labels=[label for label in concept2label.values()],
        db_relations=[qid for qid in rel2label.keys()],
        db_relation_labels=[label for label in rel2label.values()],
        db_entities=[ent for ent in ex["entities"]],
        db_entity_labels=[ent2label.get(ent, "") for ent in ex["entities"]] if ent2label else [],
        schema_serialization_type=data_training_args.schema_serialization_type,
        schema_serialization_with_db_id=data_training_args.schema_serialization_with_db_id,
        schema_serialization_with_db_content=data_training_args.schema_serialization_with_db_content,
        schema_serialization_with_db_entity_content=data_training_args.schema_serialization_with_db_entity_content,
        schema_serialization_with_db_relation_content=data_training_args.schema_serialization_with_db_relation_content
    )
    return {"serialized_schema": serialized_schema}

def add_whitespace(content):
    new_content = content.replace("ns:", "ns: ").replace(
        "rdf:", "rdf: ").replace(
        "rdfs:", "rdfs: ")
    return new_content

def remove_whitespace(content):
    new_content = content.replace("ns: ", "ns:").replace(
        "rdf: ", "rdf:").replace(
        "rdfs: ", "rdfs:")
    return new_content


def cwq_pre_process_function(
        batch: dict,
        mode: Optional[str],
        data_training_args: DataTrainingArguments,
        tokenizer: PreTrainedTokenizerBase,
) -> dict:
    prefix = data_training_args.source_prefix if data_training_args.source_prefix is not None else "question: "
    if data_training_args.use_decomposition:
        inputs = []
        targets = []
        if data_training_args.stage == "content":
            eval_format_list = []
            with open(data_training_args.structure_path) as f:
                info = json.load(f)
                for item in info:
                    eval_format_list.append(item["prediction"])
            print(f"load {len(eval_format_list)} eval_formats from {data_training_args.structure_path}")

        count = 0

        for kb, qid, question, query, serialized_schema, structure_0, content_0 in zip(batch["kb"],
                                                                                   batch["uid"],
                                                                                   batch["question"],
                                                                                   batch["query"],
                                                                                   batch["serialized_schema"],
                                                                                   batch["structure"],
                                                                                   batch["content"]):
            structure = encode(structure_0)
            # add a whitespace between prefix and relation or entity
            content_1 = add_whitespace(content_0)
            content = encode(content_1)

            input_str = cwq_get_input(question=question, serialized_schema=serialized_schema, prefix=prefix)

            if data_training_args.stage == "structure":
                inputs.append(data_training_args.schema_serialization_with_prompt + ' | ' + input_str)
                target = cwq_get_target(
                    query=structure,
                    db_id=kb,
                    target_with_db_id=data_training_args.target_with_db_id,
                )
                targets.append(target)

            elif data_training_args.stage == "content":
                if mode == 'eval':
                    input_str = data_training_args.schema_serialization_with_prompt + eval_format_list[count] + ' | ' + input_str
                else:
                    input_str = data_training_args.schema_serialization_with_prompt + structure + ' | ' + input_str
                inputs.append(input_str)
                target = content
                targets.append(target)
            count += 1

    else:
        inputs = [
            cwq_get_input(question=question, serialized_schema=serialized_schema, prefix=prefix)
            for question, serialized_schema in zip(batch["question"], batch["serialized_schema"])
        ]
        targets = [
            cwq_get_target(
                query=query,
                db_id=kb,
                target_with_db_id=data_training_args.target_with_db_id,
            )
            for kb, query in zip(batch["kb"], batch["query"])
        ]
    print(f"{mode}: {len(inputs)}")

    model_inputs: dict = tokenizer(
        inputs,
        # max_length=max_source_length,
        pad_to_multiple_of=8,
        padding=True,
        truncation=True,
        return_overflowing_tokens=False,
    )

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            # max_length=max_source_length,
            pad_to_multiple_of=8,
            padding=True,
            truncation=True,
            return_overflowing_tokens=False,
        )

    print("input: \n", inputs[0])
    print("target: \n", targets[0])
    for model_input, label_id in zip(model_inputs["input_ids"], labels["input_ids"]):
        input_tokens = tokenizer.convert_ids_to_tokens(model_input)
        label_tokens = tokenizer.convert_ids_to_tokens(label_id)
        # print(input_tokens)
        # print(label_tokens)

    if data_training_args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


class CWQTrainer(Seq2SeqTrainer):
    def _post_process_function(
            self, examples: Dataset, features: Dataset, predictions: np.ndarray, stage: str
    ) -> EvalPrediction:
        inputs = self.tokenizer.batch_decode([f["input_ids"] for f in features], skip_special_tokens=True)

        predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        label_ids = []
        for f in features:
            label_id = np.array(f["labels"])
            if self.ignore_pad_token_for_loss:
                # Replace -100 in the labels as we can't decode them.
                label_id = np.where(label_id != -100, label_id, self.tokenizer.pad_token_id)

            label_ids.append(label_id)

        decoded_label_ids = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        metas = [
            {
                "uid": example["uid"],
                "question": example["question"],
                "input": input_,
                "query": example["query"],
                "structure": example["structure"],
                "content": example["content"],
                "label": label_
            }
            for example, input_, label_ in zip(examples, inputs, decoded_label_ids)
        ]

        for i, p, d in zip(inputs[0:5], predictions[0:5], decoded_label_ids[0:5]):
            print("inpt: ", i)
            print("pred: ", p)
            print("deco: ", d)
            print("\n")

        if self.stage == "content":
            final_pred_sparqls = []
            hypotheses_path = os.path.join(self.args.output_dir, "hypotheses.json")
            if os.path.exists(hypotheses_path):
                # sentence-level check
                with open(hypotheses_path) as f:
                    hypotheses = json.load(f)
                    for idx, item in enumerate(hypotheses):
                        db_id, structure = item["structure"].split(" | ")
                        final_pred_sparql = None
                        for hypothesis in item["topk_preds"]:
                            try:
                                # combine structure and content to the final sparql query
                                final_pred_sparql = decode(combine_SC(content=remove_whitespace(hypothesis), structure=structure))
                                break
                            except:
                                continue
                        if final_pred_sparql == None:
                            # default to the first one
                            final_pred_sparql = decode(combine_SC(content=remove_whitespace(item["topk_preds"][0]), structure=structure))
                        final_pred_sparqls.append(final_pred_sparql)

                os.remove(hypotheses_path)

            else:
                for pred_content, meta in zip(predictions, metas):
                    final_pred_sparqls.append(decode(combine_SC(content=remove_whitespace(pred_content), structure=meta['structure'])))

            for pred_sparql, meta in zip(final_pred_sparqls, metas):
                meta.update({"pred_sparql": pred_sparql})

            # write predicted sparql
            with open(f"{self.args.output_dir}/predicted_sparql.txt", "w") as f:
                for final_pred_sparql in final_pred_sparqls:
                    f.write(final_pred_sparql+"\n")

            with open(f"{self.args.output_dir}/content.json", "w") as f:
                json.dump(
                    [dict(**{"uid": meta["uid"]}, **{"input": meta["input"]}, **{"prediction": prediction},
                          **{"label": label}, **{"score": prediction == label}, **{"pred_sparql": final_pred_sparql},
                          **{"gold_sparql": meta["query"]})
                     for meta, prediction, final_pred_sparql, label in
                     zip(metas, predictions, final_pred_sparqls, decoded_label_ids)],
                    f,
                    indent=4
                )
            return EvalPrediction(predictions=predictions, label_ids=decoded_label_ids, metas=metas)
        elif self.stage == "structure":
            return EvalPrediction(predictions=predictions, label_ids=decoded_label_ids, metas=metas)


    def _compute_metrics(self, eval_prediction: EvalPrediction) -> dict:
        predictions, label_ids, metas = eval_prediction

        eval_metric = self.metric.compute(predictions=predictions, references=label_ids)["exact_match"]

        if self.stage == "structure":
            if eval_metric >= self.best_acc:
                with open(f"{self.args.output_dir}/structure.json", "w") as f:
                    json.dump(
                        [dict(**{"uid": meta["uid"]}, **{"input": meta["input"]}, **{"prediction": prediction},
                              **{"label": label}, **{"score": prediction == label})
                         for meta, prediction, label in zip(metas, predictions, label_ids)],
                        f,
                        indent=4
                    )
            return {**{"eval_exact_match": eval_metric}}
        elif self.stage == "content":
            if eval_metric >= self.best_acc:
                with open(f"{self.args.output_dir}/content.json", "w") as f:
                    json.dump(
                        [dict(**{"uid": meta["uid"]}, **{"input": meta["input"]}, **{"prediction": prediction},
                              **{"label": label}, **{"score": prediction == label},
                              **{"pred_sparql": meta["pred_sparql"]},
                              **{"gold_sparql": meta["query"]})
                         for meta, prediction, label in zip(metas, predictions, label_ids)],
                        f,
                        indent=4
                    )

            final_pred_sparqls = [meta["pred_sparql"] for meta in metas]
            gold_sparqls = [meta["query"] for meta in metas]
            eval_metric_sparql = self.metric.compute(predictions=final_pred_sparqls, references=gold_sparqls)[
                "exact_match"]

            return {**{"eval_exact_match": eval_metric, "eval_exact_match_sparql": eval_metric_sparql}}


