import sys
from copy import deepcopy
from typing import Optional, Union, Any, Callable, AsyncContextManager, List, Dict, Iterable, List, Tuple, Union
import inspect
import subprocess
import asyncio
import json
import os
import pprint
import torch
import warnings
from tenacity import retry, wait_random_exponential, stop_after_delay, before_sleep_log
from transformers import LogitsProcessorList, StoppingCriteriaList, Constraint, BeamScorer, BeamSearchScorer, \
    ConstrainedBeamSearchScorer
from transformers.generation_logits_process import LogitsProcessor, LOGITS_PROCESSOR_INPUTS_DOCSTRING
from transformers.utils import add_start_docstrings
from transformers.configuration_utils import PretrainedConfig
from transformers.generation_utils import GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput, \
    BeamSearchDecoderOnlyOutput, BeamSearchEncoderDecoderOutput
from transformers.file_utils import copy_func
from transformers.models.auto import AutoModelForSeq2SeqLM
from transformers.models.auto.auto_factory import _get_model_class
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.pytorch_utils import torch_int_div
import torch.nn as nn
import math
import logging
import time
from .transformers_cfg.grammar_utils import VanillaGrammarConstraint

logger = logging.getLogger(__name__)


def with_grammar(
        model_cls: AutoModelForSeq2SeqLM,
        tokenizer: PreTrainedTokenizerFast,
        grammar_constraint: VanillaGrammarConstraint,
        stage: str = 'content',
        subgraphs: Optional[Dict[str, dict]] = None,
):

    subgraphs_eval = subgraphs["eval"]
    subgraph_vocabs = []
    subgraphs_cache = []
    for sg_k, sg_v in subgraphs_eval.items():
        subgraph = "relations: " + sg_v
        schema_term = tokenizer.convert_ids_to_tokens(tokenizer(subgraph)["input_ids"], skip_special_tokens=True)
        subgraph_vocabs.append(schema_term)
        subgraphs_cache.append(sg_v.split(", "))


    @torch.no_grad()
    def _generate(
            self,
            inputs: Optional[torch.Tensor] = None,
            max_length: Optional[int] = None,
            min_length: Optional[int] = None,
            do_sample: Optional[bool] = None,
            early_stopping: Optional[bool] = None,
            num_beams: Optional[int] = None,
            temperature: Optional[float] = None,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            typical_p: Optional[float] = None,
            repetition_penalty: Optional[float] = None,
            bad_words_ids: Optional[Iterable[int]] = None,
            bos_token_id: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
            length_penalty: Optional[float] = None,
            no_repeat_ngram_size: Optional[int] = None,
            encoder_no_repeat_ngram_size: Optional[int] = None,
            num_return_sequences: Optional[int] = None,
            max_time: Optional[float] = None,
            max_new_tokens: Optional[int] = None,
            decoder_start_token_id: Optional[int] = None,
            use_cache: Optional[bool] = None,
            num_beam_groups: Optional[int] = None,
            diversity_penalty: Optional[float] = None,
            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
            logits_processor: Optional[LogitsProcessorList] = LogitsProcessorList(),
            renormalize_logits: Optional[bool] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = StoppingCriteriaList(),
            constraints: Optional[List[Constraint]] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_scores: Optional[bool] = None,
            return_dict_in_generate: Optional[bool] = None,
            forced_bos_token_id: Optional[int] = None,
            forced_eos_token_id: Optional[int] = None,
            remove_invalid_values: Optional[bool] = None,
            synced_gpus: Optional[bool] = False,
            exponential_decay_length_penalty: Optional[Tuple[Union[int, float]]] = None,
            structures: Optional[List[str]] = None,
            ontologies: Optional[List[str]] = None,
            observed_num_examples: Optional[int] = None,
            **model_kwargs,
    ) -> Union[GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput, torch.LongTensor]:
        r"""
        Generates sequences for models with a language modeling head. The method currently supports greedy decoding,
        multinomial sampling, beam-search decoding, and beam-search multinomial sampling.

        Apart from `inputs`, all the arguments below will default to the value of the attribute of the same name inside
        the [`PretrainedConfig`] of the model. The default values indicated are the default values of those config.

        Most of these parameters are explained in more detail in [this blog
        post](https://huggingface.co/blog/how-to-generate).

        Parameters:
            inputs (`torch.Tensor` of shape `(batch_size, sequence_length)`, `(batch_size, sequence_length,
            feature_dim)` or `(batch_size, num_channels, height, width)`, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should of in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            max_length (`int`, *optional*, defaults to `model.config.max_length`):
                The maximum length of the sequence to be generated.
            max_new_tokens (`int`, *optional*, defaults to None):
                The maximum numbers of tokens to generate, ignore the current number of tokens. Use either
                `max_new_tokens` or `max_length` but not both, they serve the same purpose.
            min_length (`int`, *optional*, defaults to 10):
                The minimum length of the sequence to be generated.
            do_sample (`bool`, *optional*, defaults to `False`):
                Whether or not to use sampling ; use greedy decoding otherwise.
            early_stopping (`bool`, *optional*, defaults to `False`):
                Whether to stop the beam search when at least `num_beams` sentences are finished per batch or not.
            num_beams (`int`, *optional*, defaults to 1):
                Number of beams for beam search. 1 means no beam search.
            temperature (`float`, *optional*, defaults to 1.0):
                The value used to module the next token probabilities.
            top_k (`int`, *optional*, defaults to 50):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (`float`, *optional*, defaults to 1.0):
                If set to float < 1, only the most probable tokens with probabilities that add up to `top_p` or higher
                are kept for generation.
            repetition_penalty (`float`, *optional*, defaults to 1.0):
                The parameter for repetition penalty. 1.0 means no penalty. See [this
                paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            bos_token_id (`int`, *optional*):
                The id of the *beginning-of-sequence* token.
            eos_token_id (`int`, *optional*):
                The id of the *end-of-sequence* token.
            length_penalty (`float`, *optional*, defaults to 1.0):
                Exponential penalty to the length. 1.0 means no penalty. Set to values < 1.0 in order to encourage the
                model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer
                sequences.
            no_repeat_ngram_size (`int`, *optional*, defaults to 0):
                If set to int > 0, all ngrams of that size can only occur once.
            encoder_no_repeat_ngram_size (`int`, *optional*, defaults to 0):
                If set to int > 0, all ngrams of that size that occur in the `encoder_input_ids` cannot occur in the
                `decoder_input_ids`.
            bad_words_ids(`List[List[int]]`, *optional*):
                List of token ids that are not allowed to be generated. In order to get the token ids of the words that
                should not appear in the generated text, use `tokenizer(bad_words, add_prefix_space=True,
                add_special_tokens=False).input_ids`.
            num_return_sequences(`int`, *optional*, defaults to 1):
                The number of independently computed returned sequences for each element in the batch.
            max_time(`float`, *optional*, defaults to None):
                The maximum amount of time you allow the computation to run for in seconds. generation will still
                finish the current pass after allocated time has been passed.
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values are in `[0, 1]`, 1 for tokens
                that are not masked, and 0 for masked tokens. If not provided, will default to a tensor the same shape
                as `input_ids` that masks the pad token. [What are attention masks?](../glossary#attention-mask)
            decoder_start_token_id (`int`, *optional*):
                If an encoder-decoder model starts decoding with a different token than *bos*, the id of that token.
            use_cache: (`bool`, *optional*, defaults to `True`):
                Whether or not the model should use the past last key/values attentions (if applicable to the model) to
                speed up decoding.
            num_beam_groups (`int`, *optional*, defaults to 1):
                Number of groups to divide `num_beams` into in order to ensure diversity among different groups of
                beams. [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.
            diversity_penalty (`float`, *optional*, defaults to 0.0):
                This value is subtracted from a beam's score if it generates a token same as any beam from other group
                at a particular time. Note that `diversity_penalty` is only effective if `group beam search` is
                enabled.
            prefix_allowed_tokens_fn: (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
                `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
                on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
                for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://arxiv.org/abs/2010.00904).
            logits_processor (`LogitsProcessorList`, *optional*):
                 Custom logits processors that complement the default logits processors built from arguments and a
                 model's config. If a logit processor is passed that is already created with the arguments or a model's
                 config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                 Custom stopping criteria that complement the default stopping criteria built from arguments and a
                 model's config. If a stopping criteria is passed that is already created with the arguments or a
                 model's config an error is thrown. This feature is intended for advanced users.
            constraints (`List[Constraint]`, *optional*):
                 Custom constraints that can be added to the generation to ensure that the output will contain the use
                 of certain tokens as defined by `Constraint` objects, in the most sensible way possible.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
            forced_bos_token_id (`int`, *optional*):
                The id of the token to force as the first generated token after the `decoder_start_token_id`. Useful
                for multilingual models like [mBART](../model_doc/mbart) where the first generated token needs to be
                the target language token.
            forced_eos_token_id (`int`, *optional*):
                The id of the token to force as the last generated token when `max_length` is reached.
            remove_invalid_values (`bool`, *optional*):
                Whether to remove possible *nan* and *inf* outputs of the model to prevent the generation method to
                crash. Note that using `remove_invalid_values` can slow down generation.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If the model
                is an encoder-decoder model, encoder specific kwargs should not be prefixed and decoder specific kwargs
                should be prefixed with *decoder_*.

        Return:
            [`~file_utils.ModelOutput`] or `torch.LongTensor`: A [`~file_utils.ModelOutput`] (if
            `return_dict_in_generate=True` or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.

                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                [`~file_utils.ModelOutput`] types are:

                    - [`~generation_utils.GreedySearchDecoderOnlyOutput`],
                    - [`~generation_utils.SampleDecoderOnlyOutput`],
                    - [`~generation_utils.BeamSearchDecoderOnlyOutput`],
                    - [`~generation_utils.BeamSampleDecoderOnlyOutput`]

                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~file_utils.ModelOutput`] types are:

                    - [`~generation_utils.GreedySearchEncoderDecoderOutput`],
                    - [`~generation_utils.SampleEncoderDecoderOutput`],
                    - [`~generation_utils.BeamSearchEncoderDecoderOutput`],
                    - [`~generation_utils.BeamSampleEncoderDecoderOutput`]

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

        >>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        >>> # do greedy decoding without providing a prompt
        >>> outputs = model.generate(max_length=40)
        >>> print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        >>> document = (
        ...     "at least two people were killed in a suspected bomb attack on a passenger bus "
        ...     "in the strife-torn southern philippines on monday , the military said."
        ... )
        >>> # encode input context
        >>> input_ids = tokenizer(document, return_tensors="pt").input_ids
        >>> # generate 3 independent sequences using beam search decoding (5 beams)
        >>> # with T5 encoder-decoder model conditioned on short news article.
        >>> outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=3)
        >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))

        >>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        >>> input_context = "The dog"
        >>> # encode input context
        >>> input_ids = tokenizer(input_context, return_tensors="pt").input_ids
        >>> # generate 3 candidates using sampling
        >>> outputs = model.generate(input_ids=input_ids, max_length=20, num_return_sequences=3, do_sample=True)
        >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))

        >>> tokenizer = AutoTokenizer.from_pretrained("ctrl")
        >>> model = AutoModelForCausalLM.from_pretrained("ctrl")
        >>> # "Legal" is one of the control codes for ctrl
        >>> input_context = "Legal My neighbor is"
        >>> # encode input context
        >>> input_ids = tokenizer(input_context, return_tensors="pt").input_ids
        >>> outputs = model.generate(input_ids=input_ids, max_length=20, repetition_penalty=1.2)
        >>> print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))

        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=False)
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> input_context = "My cute dog"
        >>> # get tokens of words that should not be generated
        >>> bad_words_ids = tokenizer(
        ...     ["idiot", "stupid", "shut up"], add_prefix_space=True, add_special_tokens=False
        >>> ).input_ids
        >>> # encode input context
        >>> input_ids = tokenizer(input_context, return_tensors="pt").input_ids
        >>> # generate sequences without allowing bad_words to be generated
        >>> outputs = model.generate(input_ids=input_ids, max_length=20, do_sample=True, bad_words_ids=bad_words_ids)
        >>> print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))
        ```"""
        # 1. Set generation parameters if not already defined
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
        num_beam_groups = num_beam_groups if num_beam_groups is not None else self.config.num_beam_groups
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )

        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        if eos_token_id is None and hasattr(self.config, "decoder"):
            eos_token_id = self.config.decoder.eos_token_id

        if pad_token_id is None and eos_token_id is not None:
            # special case if pad_token_id is not defined
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            pad_token_id = eos_token_id

        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # 2. Define model inputs
        # inputs_tensor has to be defined
        # model_input_name is defined if model-specific keyword input is passed
        # otherwise model_input_name is None
        # all model-specific keyword inputs are removed from `model_kwargs`
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(inputs, bos_token_id, model_kwargs)
        batch_size = inputs_tensor.shape[0]

        # 3. Define other model kwargs
        model_kwargs["output_attentions"] = output_attentions
        model_kwargs["output_hidden_states"] = output_hidden_states
        model_kwargs["use_cache"] = use_cache

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs

        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, pad_token_id, eos_token_id
            )

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created
            # and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name
            )

        # 4. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids = self._prepare_decoder_input_ids_for_generation(
                batch_size,
                decoder_start_token_id=decoder_start_token_id,
                bos_token_id=bos_token_id,
                model_kwargs=model_kwargs,
            )
        else:
            # if decoder-only then inputs_tensor has to be `input_ids`
            input_ids = inputs_tensor

        # 5. Prepare `max_length` depending on other stopping criteria
        # if `max_new_tokens` is passed, but not `max_length` -> set `max_length = max_new_tokens`
        input_ids_seq_length = input_ids.shape[-1]
        if max_length is None and max_new_tokens is not None:
            max_length = max_new_tokens + input_ids_seq_length
        elif max_length is not None and max_new_tokens is not None:
            # Both are set, this is odd, raise a warning
            warnings.warn(
                "Both `max_length` and `max_new_tokens` have been set "
                f"but they serve the same purpose. `max_length` {max_length} "
                f"will take priority over `max_new_tokens` {max_new_tokens}.",
                UserWarning,
            )
        # default to config if still None
        max_length = max_length if max_length is not None else self.config.max_length

        if input_ids_seq_length >= max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids_seq_length}, but ``max_length`` is set to {max_length}. "
                "This can lead to unexpected behavior. You should consider increasing ``config.max_length`` or ``max_length``."
            )

        # 6. determine generation mode
        is_constraint_gen_mode = constraints is not None
        is_greedy_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is False and constraints is None
        is_sample_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is True and constraints is None
        is_beam_gen_mode = (num_beams > 1) and (num_beam_groups == 1) and do_sample is False and constraints is None
        is_beam_sample_gen_mode = (
                (num_beams > 1) and (num_beam_groups == 1) and do_sample is True and constraints is None
        )
        is_group_beam_gen_mode = (num_beams > 1) and (num_beam_groups > 1) and constraints is None

        if num_beam_groups > num_beams:
            raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")
        if is_group_beam_gen_mode and do_sample is True:
            raise ValueError(
                "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
            )

        # 7. prepare distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=inputs_tensor,
            bad_words_ids=bad_words_ids,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            remove_invalid_values=remove_invalid_values,
            exponential_decay_length_penalty=exponential_decay_length_penalty,
            logits_processor=logits_processor,
            renormalize_logits=renormalize_logits,
        )
        logits_processor.append(
            GrammarConstrainedLogitsProcessor(
                eos_token_id=eos_token_id,
                max_tokens_to_check=10,
                tokenizer=tokenizer,
                stage=stage,
                grammar_constraint=grammar_constraint,
                observed_num_examples=observed_num_examples,
                subgraph_vocabs=subgraph_vocabs,
                subgraphs=subgraphs_cache,
                parse_start_index=1
            )
        )

        # 8. prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            max_length=max_length, max_time=max_time, stopping_criteria=stopping_criteria
        )

        # 9. go into different generation modes
        if is_greedy_gen_mode:
            if num_return_sequences > 1:
                raise ValueError(
                    f"num_return_sequences has to be 1, but is {num_return_sequences} when doing greedy search."
                )

            # 10. run greedy search
            return self.greedy_search(
                input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_sample_gen_mode:
            # 10. prepare logits warper
            logits_warper = self._get_logits_warper(
                top_k=top_k, top_p=top_p, typical_p=typical_p, temperature=temperature, num_beams=num_beams,
                renormalize_logits=renormalize_logits
            )

            # 11. expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids,
                expand_size=num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 12. run sample
            return self.sample(
                input_ids,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_beam_gen_mode:
            if num_return_sequences > num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            # 10. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
            )
            # 11. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids, expand_size=num_beams, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs
            )
            # 12. run beam search
            return self.beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                structures=structures,
                ontologies=ontologies,
                observed_num_examples=observed_num_examples,
                **model_kwargs,
            )

        elif is_beam_sample_gen_mode:
            # 10. prepare logits warper
            logits_warper = self._get_logits_warper(
                top_k=top_k, top_p=top_p, typical_p=typical_p, temperature=temperature, num_beams=num_beams
            )

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")
            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size * num_return_sequences,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
            )

            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids,
                expand_size=num_beams * num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 13. run beam sample
            return self.beam_sample(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_group_beam_gen_mode:
            if num_return_sequences > num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if num_beams % num_beam_groups != 0:
                raise ValueError("`num_beams` should be divisible by `num_beam_groups` for group beam search.")

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            # 10. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                max_length=stopping_criteria.max_length,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
                num_beam_groups=num_beam_groups,
            )
            # 11. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids, expand_size=num_beams, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs
            )
            # 12. run beam search
            return self.group_beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_constraint_gen_mode:
            if num_return_sequences > num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            if num_beams <= 1:
                raise ValueError("`num_beams` needs to be greater than 1 for constrained genertation.")

            if do_sample:
                raise ValueError("`do_sample` needs to be false for constrained generation.")

            if num_beam_groups is not None and num_beam_groups > 1:
                raise ValueError("`num_beam_groups` not supported yet for constrained generation.")

            # 10. prepare beam search scorer
            constrained_beam_scorer = ConstrainedBeamSearchScorer(
                constraints=constraints,
                batch_size=batch_size,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
            )
            # 11. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids, expand_size=num_beams, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs
            )
            # 12. run beam search
            return self.constrained_beam_search(
                input_ids,
                constrained_beam_scorer=constrained_beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

    def _beam_search(
            self,
            input_ids: torch.LongTensor,
            beam_scorer: BeamScorer,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            max_length: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_scores: Optional[bool] = None,
            return_dict_in_generate: Optional[bool] = None,
            synced_gpus: Optional[bool] = False,
            structures: Optional[List[str]] = None,
            ontologies: Optional[List[str]] = None,
            observed_num_examples: Optional[int] = None,
            **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:
        r"""
        Generates sequences for models with a language modeling head using beam search decoding.

        Parameters:

            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            beam_scorer (`BeamScorer`):
                An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
                sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`int`, *optional*):
                The id of the *end-of-sequence* token.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`generation_utilsBeamSearchDecoderOnlyOutput`], [`~generation_utils.BeamSearchEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation_utils.BeamSearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation_utils.BeamSearchEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.


        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForSeq2SeqLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     BeamSearchScorer,
        ... )
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

        >>> encoder_input_str = "translate English to German: How old are you?"
        >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids


        >>> # lets run beam search using 3 beams
        >>> num_beams = 3
        >>> # define decoder start token ids
        >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
        >>> input_ids = input_ids * model.config.decoder_start_token_id

        >>> # add encoder_outputs to model keyword arguments
        >>> model_kwargs = {
        ...     "encoder_outputs": model.get_encoder()(
        ...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
        ...     )
        ... }

        >>> # instantiate beam scorer
        >>> beam_scorer = BeamSearchScorer(
        ...     batch_size=1,
        ...     num_beams=num_beams,
        ...     device=model.device,
        ... )

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
        ...     ]
        ... )

        >>> outputs = model.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs)

        >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
        ```"""
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False  # used by synced_gpus only
        while True:

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]
            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `nn.functional.log_softmax` operation.
            next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed = logits_processor(input_ids, next_token_scores, structures=structures, ontologies=ontologies, observed_num_examples=observed_num_examples)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = torch_int_div(next_tokens, vocab_size)
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], beam_idx)

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None
            else:
                num_return_sequences = beam_scorer.num_beam_hyps_to_keep
                # return only as many indices as sequences
                beam_indices = tuple(
                    (beam_indices[i * num_beams: i * num_beams + num_return_sequences] for i in range(batch_size))
                )
                beam_indices = sum(beam_indices, ())

            if self.config.is_encoder_decoder:
                return BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=beam_indices,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=beam_indices,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs["sequences"]

    class _AutoModelClass(model_cls):
        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
            config = kwargs.pop("config", None)
            kwargs["_from_auto"] = True
            if not isinstance(config, PretrainedConfig):
                config, kwargs = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path, return_unused_kwargs=True, **kwargs
                )

            if type(config) in cls._model_mapping.keys():
                model_class = _get_model_class(config, cls._model_mapping)
                generate = copy_func(_generate)
                beam_search = copy_func(_beam_search)
                generate.__doc__ = model_class.generate.__doc__
                model_class.old_generate = copy_func(model_class.generate)
                model_class.generate = generate
                model_class.beam_search = beam_search
                # model_class.add_schema = staticmethod(copy_func(_add_schema))
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
            raise ValueError(
                f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
                f"Model type should be one of {', '.join(c.__name__ for c in cls._model_mapping.keys())}."
            )

    return _AutoModelClass


class GrammarConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(self,
                 grammar_constraint: VanillaGrammarConstraint,
                 stage: str,
                 max_tokens_to_check: int,
                 tokenizer: PreTrainedTokenizerFast,
                 eos_token_id: int,
                 observed_num_examples: int,
                 subgraph_vocabs=None,
                 subgraphs=None,
                 parse_start_index=None):
        self.stage = stage
        self.max_tokens_to_check = max_tokens_to_check
        self.last_size = None
        self.current_batch = 0
        self.grammar_constraint = grammar_constraint
        self.batch_stacks = None
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id
        self.num_beams = None
        self.parse_start_index = None
        self.observed_num_examples = observed_num_examples
        self.Structure_Vocab = {
            '<pad>', '</s>', '▁select', '▁ask', '▁distinct', '▁asterisk', '▁where', '▁filter', '▁order', '▁by', '▁limit', '▁group', '▁union',
            '▁having', '▁count', '▁as', '▁max', '▁min', '▁offset', '▁avg', '▁sum', '▁year', '▁sum', '▁asc', '▁desc', '▁not', '▁exists', '▁in',
            '▁[', 'var', ']', 'con', 're', 'l', 'ent', 'val',
            '▁bra', 'ce', 'open', 'close', '▁separat', 'or', '_', 'd', 'o', 't', 's', 'e', 'm', 'i', '▁par', 'com',
            '▁math', 'n', 'q', '_', 'g', 'boo', 'not', '▁is', 'liter', 'al', '▁or', 'logical', '▁and', '▁lang', 'match', '▁single', 'quot', 'be', 'in', 'end',
        }
        self.Ontology_Base_Vocab = {'n', 's', ':', 'ns', '▁ns', 'r', 'd', 'f', '▁rdf', '▁rdfs', 'd', 'b', 'o', '▁dbo', '_dbr', 'p', '▁dbp', '▁dbc', 'c',
                                    'x', '▁xsd', 'w', 'l', '▁owl', 'y', 'a', 'g', '▁yago', 'k', '▁skos',
                                    '<pad>', '</s>', '.', '▁', '▁[', ']', 're', 'rel', 'var', 'ent', 'con', 'val', '_',
                                       '▁separat', 'or', 't', 'close'}
        self.Subgraph_Vocabs = subgraph_vocabs
        self.Subgraphs = subgraphs

    def mask_logits(self, logits, device):
        # resolve each stack to a tensor of True/False for each token
        # indicating acceptance
        # acceptance = self.grammar_acceptor.filter_vocab(self.stacks, device)
        acceptance = self.grammar_constraint.batch_filter_vocab(
            self.batch_stacks, device
        )
        # acceptance is a tensor of shape (batch_size, vocab_size)
        # get the indices of the accepted tokens
        # do the following operation only in debug mode
        if os.getenv("DEBUG_MODE") == "True":
            # convert acceptance to numpy array
            batch_size, vocab_size = acceptance.shape
            acceptance_np = acceptance.cpu().numpy()
            accepted_x, accepted_y = acceptance_np.nonzero()
            # dict of {batch_index: [accepted_token_indices]}
            # initialize the dict with empty list
            accepted_token_indices = {i: [] for i in range(batch_size)}
            for x, y in zip(accepted_x, accepted_y):
                accepted_token_indices[x].append(y)
            logger.debug("Accepted token indices for the current batch:")
            logger.debug("\n" + pprint.pformat(accepted_token_indices))
            # convert token_ids to tokens
            accepted_tokens = {
                i: [
                    self.grammar_constraint.tokenizer.decode([token_id])
                    for token_id in token_ids
                ]
                for i, token_ids in accepted_token_indices.items()
            }
            logger.debug("Accepted tokens for the current batch:")
            logger.debug("\n" + pprint.pformat(accepted_tokens))
        # Logits to -inf where False
        logits[~acceptance] = -math.inf

    # TODO: batching
    def process_logits(self, input_ids, scores):
        """
        :param input_ids:
        :param scores:
        :return:
        """
        # we dynamically create stacks at the first call, so that we know the batch size and beam size
        if self.batch_stacks is None:
            self.batch_stacks = [
                self.grammar_constraint.init_stacks() for _ in range(len(input_ids))
            ]

        if os.getenv("DEBUG_MODE") == "True":
            print("-" * 80)

        logger.debug("input_ids: \n" + pprint.pformat(input_ids))
        logger.debug("scores: \n" + pprint.pformat(scores))
        logger.debug("last_size: \n" + pprint.pformat(self.last_size))
        logger.debug(
            "num of stacks: \n"
            + pprint.pformat([len(stack) for stack in self.batch_stacks])
        )
        logger.debug("stacks: \n" + pprint.pformat(self.batch_stacks))

        self.batch_stacks = self.grammar_constraint.advance_token_ids(
            input_ids, self.batch_stacks, self.parse_start_index
        )

        self.mask_logits(scores, scores.device)
        return scores

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor,
            structures: Optional[List[str]]=None,
            ontologies: Optional[List[str]]=None,
            observed_num_examples: Optional[int] = None,
    ) -> torch.FloatTensor:

        if not self.num_beams:
            self.num_beams = input_ids.size(0) // len(structures)

        batch_size = input_ids.size(0) // self.num_beams

        #print("batch_size: ", batch_size)

        if self.stage == "structure":
            return self.structure_constrained(input_ids, scores)
        elif self.stage == "content":

            batch_subgraph_vocabs = [
                sorted(set(batch_subgraph_vocab))
                for batch_subgraph_vocab in self.Subgraph_Vocabs[observed_num_examples - batch_size: observed_num_examples]
            ]

            batch_subgraphs = [
                batch_subgraph
                for batch_subgraph in
                self.Subgraphs[observed_num_examples - batch_size: observed_num_examples]
            ]

            return self.content_constrained(input_ids=input_ids, scores=scores, structures=structures, subgraph_vocabs=batch_subgraph_vocabs, subgraphs=batch_subgraphs, ontologies=ontologies)

    @torch.no_grad()
    def structure_constrained(
            self,
            input_ids: torch.Tensor,
            scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        #for input_id in input_ids:
        #    print(self.tokenizer.convert_ids_to_tokens(input_id))
        return self.process_logits(input_ids, scores)

    @torch.no_grad()
    def content_constrained(
            self,
            input_ids: torch.Tensor,
            scores: torch.FloatTensor,
            structures: List[str],
            subgraph_vocabs: List[set],
            subgraphs: List[List[str]],
            ontologies: List[str]
    ) -> torch.FloatTensor:

        top_k = min(max(1, 20), scores.size(-1))
        top_scores, top_token_ids = torch.topk(scores, top_k)
        for i in range(top_scores.size(0)):
            if top_scores[i][1] == -math.inf:
                if top_scores[i][0] == -math.inf:
                    top_scores[i][1] = 1
                else:
                    top_scores[i][1] = top_scores[i][0]

        # Remove all tokens with a probability less than the last token of the top-k
        lowest_top_k_scores = top_scores[..., -1, None]
        del top_scores
        indices_to_remove = scores < lowest_top_k_scores
        del lowest_top_k_scores
        # Do not mask the EOS token because otherwise production can continue indefinitely if all other tokens are masked
        indices_to_remove[:, self.eos_token_id] = False

        #print(self.tokenizer.convert_tokens_to_ids(['[']))
        predictions = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        top_tokens = [self.tokenizer.convert_ids_to_tokens(top_token_id) for top_token_id in top_token_ids.tolist()]
        structure_token_ids = self.tokenizer(structures)["input_ids"]
        structure_tokens = [self.tokenizer.convert_ids_to_tokens(structure_token_id, skip_special_tokens=True) for structure_token_id in structure_token_ids]

        entities = []
        for onto in ontologies:

            ents = []
            ent_string = onto[onto.index('entities: ')+10:]

            if ent_string.count('(') == 0:
                ent = ent_string[:-1]
                ents.append(ent)
            elif ent_string.count('(') == 1:
                try:
                    ent = ent_string[:ent_string.index(' (')]
                    ents.append(ent)
                except:
                    ents.append('')
            else:
                for eee in ent_string.split(', '):
                    try:
                        ent = eee[:eee.index(' (')]
                        ents.append(ent)
                    except:
                        ents.append('')

            entities.append(ents)

        need_check_preds = [] # store the predictions that will be extended in the next step
        need_check_input_ids = []
        need_check_flags = []

        check_match_flags = []

        ## 检测pred与structure是否匹配

        for idx0, pred in enumerate(predictions):
            if len(pred) > 0 and (pred.count('[') > 0):
                structure = structures[idx0 // self.num_beams]
                if (pred.count('[') == structure.count('[')) and (pred.count(']') == structure.count(']')):
                    check_match_flags.append(True)
                else:
                    check_match_flags.append(False)
            else:
                check_match_flags.append(False)

        for idx0, pred in enumerate(predictions):

            if len(pred) > 0 and (pred.count('[') > 0):
                need_check_preds.append(pred)
                need_check_input_ids.append([0] + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(pred)))
                need_check_flags.append(True)
            else:
                indices_to_remove[idx0, :] = True
                indices_to_remove[idx0, [self.eos_token_id, 784]] = False
                need_check_flags.append(False)


        if len(need_check_preds) > 0:
            # max_len = max([len(x) for x in need_check_input_ids])
            # need_check_input_ids = [[0] * (max_len - len(x)) + x for x in need_check_input_ids]
            # need_check_input_ids = torch.tensor(need_check_input_ids)
            # check_indices_to_remove = indices_to_remove[need_check_flags, :]
            # check_top_token_ids = top_token_ids[need_check_flags, :]

            asyncio.run(
                # self._batch_mask_top_k(indices_to_remove=check_indices_to_remove,
                #                        #input_ids=need_check_input_ids,
                #                        predictions=need_check_preds,
                #                        top_tokens=check_top_tokens,
                #                        top_token_ids=check_top_token_ids,
                #                        subgraph_vocabs=subgraph_vocabs,
                #                        subgraphs=subgraphs,
                #                        structures=structures,
                #                        structure_tokens=structure_tokens,
                #                        structure_token_ids=structure_token_ids,
                #                        check_match_flags=check_match_flags
                #         ),
                self._mask_top_k(indices_to_remove=indices_to_remove,
                                 predictions=predictions,
                                 top_tokens=top_tokens,
                                 top_token_ids=top_token_ids,
                                 subgraphs=subgraphs,
                                 subgraph_vocabs=subgraph_vocabs,
                                 structures=structures,
                                 entities=entities,
                                 structure_tokens=structure_tokens,
                                 structure_token_ids=structure_token_ids,
                                 check_match_flags=check_match_flags
                                 ),
                debug=False
            )

        scores = scores.masked_fill(indices_to_remove, -float("Inf"))
        del indices_to_remove
        return scores


    async def _mask_top_k(self,
                          indices_to_remove: torch.Tensor,
                          predictions: List[str],
                          top_tokens: List[List[str]],
                          top_token_ids: torch.Tensor,
                          subgraph_vocabs: List[set],
                          subgraphs: List[List[str]],
                          structures: List[str],
                          entities: List[List[str]],
                          structure_tokens: List[List[str]],
                          structure_token_ids: List[List[int]],
                          check_match_flags: List[bool]
                          ) -> None:

        futures = [
            self._mask(
                indices_to_remove=indices_to_remove,
                batch_idx=batch_idx,
                prediction=prediction,
                structures=structures,
                batch_structure_tokens=structure_tokens,
                batch_structure_token_ids=structure_token_ids,
                batch_top_tokens=batch_top_tokens,
                batch_top_token_ids=batch_top_token_ids,
                subgraphs=subgraphs,
                subgraph_vocabs=subgraph_vocabs,
                entities=entities,
                check_match_flags=check_match_flags
            )
            for batch_idx, (prediction, batch_top_tokens, batch_top_token_ids) in enumerate(zip(predictions, top_tokens, top_token_ids))
        ]

        for f in asyncio.as_completed(futures):
            await f


    @retry(
        wait=wait_random_exponential(multiplier=1, max=60),
        stop=stop_after_delay(600),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def _mask(self,
              indices_to_remove: torch.Tensor,
              batch_idx: int,
              prediction: str,
              structures: List[str],
              entities: List[List[str]],
              batch_structure_tokens: List[List[str]],
              batch_structure_token_ids: List[List[int]],
              batch_top_tokens: List[str],
              batch_top_token_ids: torch.Tensor,
              subgraph_vocabs: List[set],
              subgraphs: List[List[str]],
              check_match_flags: [bool]
              ) -> None:

        structure = structures[batch_idx // self.num_beams]
        structure_tokens = batch_structure_tokens[batch_idx // self.num_beams]
        structure_token_ids = batch_structure_token_ids[batch_idx // self.num_beams]
        entity_ss = entities[batch_idx // self.num_beams]
        subgraph_vocab = subgraph_vocabs[batch_idx // self.num_beams]
        full_subgraph_vocab = self.Ontology_Base_Vocab.union(set(subgraph_vocab))
        full_subgraph_vocab_id = torch.tensor(self.tokenizer.convert_tokens_to_ids(list(full_subgraph_vocab)),
                                              dtype=torch.int32)
        subgraph = subgraphs[batch_idx // self.num_beams]
        check_match_flag = check_match_flags[batch_idx // self.num_beams]

        await self._check_token(indices_to_remove, batch_idx, prediction, structure, batch_top_tokens, batch_top_token_ids,
                          full_subgraph_vocab, subgraph, structure_tokens, structure_token_ids, check_match_flag, entity_ss)

        print(f"{batch_idx}: {prediction}")


    async def _check_token(self,
                     indices_to_remove: torch.Tensor,
                     batch_idx: int,
                     prediction: str,
                     structure: str,
                     top_tokens: List[str],
                     top_token_ids: torch.Tensor,
                     subgraph_vocab: set,
                     subgraph: List[str],
                     structure_tokens: List[str],
                     structure_token_ids: List[int],
                     check_match_flag: bool,
                           entity_ss: List[str]
                     ) -> None:

        if prediction.count('[') == prediction.count(']'):

            last_s_idx = prediction.rindex('[')
            last_e_idx = prediction.rindex(']')

            if prediction[last_s_idx:last_e_idx + 1] == "[rel]":
                pred_last_idx = prediction.rindex(']')
                pred_rel = prediction[pred_last_idx + 1:]
                #print("curr relation: ", pred_rel)

                if len(pred_rel.strip()) > 100:
                    indices_to_remove[batch_idx, :] = True
                    if check_match_flag:
                        indices_to_remove[batch_idx, self.eos_token_id] = False
                    else:
                        indices_to_remove[batch_idx, 784] = False
                elif pred_rel == "" or pred_rel.strip() == "" or ":" not in pred_rel:
                    pass
                elif pred_rel in subgraph:
                    indices_to_remove[batch_idx, :] = True
                    if check_match_flag:
                        indices_to_remove[batch_idx, self.eos_token_id] = False
                    else:
                        indices_to_remove[batch_idx, 784] = False
                else:
                    to_remove_count = 0
                    for top_tok, top_tok_id in zip(top_tokens, top_token_ids):
                        to_remove = True
                        to_remove_count += 1

                        if (top_tok_id.item() == 784) and not check_match_flag:
                            to_remove = False
                            to_remove_count -= 1
                        elif (top_tok_id.item() == 1) and check_match_flag:
                            to_remove = False
                            to_remove_count -= 1
                        elif top_tok in subgraph_vocab:
                            # check if each of relations in the subgraph starts with 'relation+topken'
                            top_tok_1 = self.tokenizer.batch_decode([top_tok_id], skip_special_tokens=True)[0]
                            rel = pred_rel.split(':')[1].strip() + top_tok_1
                            for sg in subgraph:
                                if sg.startswith(rel):
                                    to_remove = False
                                    to_remove_count -= 1
                                    break


                        indices_to_remove[batch_idx, top_tok_id] = to_remove
                        #print(f"next token: {top_tok}({top_tok_id}), to_remove: {to_remove}")
                    if to_remove_count == len(top_tokens):
                        print(
                            f"No any options to choose, but not finished, activate the first top token again.")
                        indices_to_remove[batch_idx, top_token_ids[0]] = False
                        print(f"Token {top_tokens[0]}({top_token_ids[0]}), to_remove: {indices_to_remove[batch_idx, top_token_ids[0]]}. No any options to choose, but not finished, activate the first top token again.")



            elif prediction[last_s_idx:last_e_idx + 1] == "[ent]":
                pred_last_idx = prediction.rindex(']')
                pred_ent = prediction[pred_last_idx + 1:]
                #print("curr ent: ", pred_ent)

                if len(pred_ent.strip()) > 20:
                    indices_to_remove[batch_idx, :] = True
                    if check_match_flag:
                        indices_to_remove[batch_idx, self.eos_token_id] = False
                    else:
                        indices_to_remove[batch_idx, 784] = False
                elif pred_ent == "" or pred_ent.strip() == "" or ":" not in pred_ent:
                    pass
                elif pred_ent in entity_ss:
                    indices_to_remove[batch_idx, :] = True
                    if check_match_flag:
                        indices_to_remove[batch_idx, self.eos_token_id] = False
                    else:
                        indices_to_remove[batch_idx, 784] = False
                else:
                    to_remove_count = 0
                    for top_tok, top_tok_id in zip(top_tokens, top_token_ids):
                        to_remove = True
                        to_remove_count += 1

                        if (top_tok_id.item() == 784) and not check_match_flag:
                            to_remove = False
                            to_remove_count -= 1
                        elif (top_tok_id.item() == 1) and check_match_flag:
                            to_remove = False
                            to_remove_count -= 1
                        else:
                            # check if each of relations in the subgraph starts with 'relation+topken'
                            top_tok_1 = self.tokenizer.batch_decode([top_tok_id], skip_special_tokens=True)[0]
                            ent = pred_ent.split(':')[1].strip() + top_tok_1
                            for entity in entity_ss:
                                if entity.startswith(ent):
                                    to_remove = False
                                    to_remove_count -= 1
                                    break

                        indices_to_remove[batch_idx, top_tok_id] = to_remove
                        #print(f"next token: {top_tok}({top_tok_id}), to_remove: {to_remove}")
                    if to_remove_count == len(top_tokens):
                        print(
                            f"No any options to choose, but not finished, activate the first top token again.")
                        indices_to_remove[batch_idx, top_token_ids[0]] = False
                        print(f"Token {top_tokens[0]} ({top_token_ids[0]}), to_remove: {indices_to_remove[batch_idx, top_token_ids[0]]}. ")




            elif prediction[last_s_idx:last_e_idx + 1] == "[con]":
                pred_last_idx = prediction.rindex(']')
                pred_con = prediction[pred_last_idx + 1:]
                # print("curr con: ", pred_con)
                if len(pred_con.strip()) > 768:
                    indices_to_remove[batch_idx, :] = True
                    if check_match_flag:
                        indices_to_remove[batch_idx, self.eos_token_id] = False
                    else:
                        indices_to_remove[batch_idx, 784] = False



            elif prediction[last_s_idx:last_e_idx + 1] == "[val]":
                pred_last_idx = prediction.rindex(']')
                pred_val = prediction[pred_last_idx + 1:]

                if len(pred_val) > 10:
                    indices_to_remove[batch_idx, :] = True
                    if check_match_flag:
                        indices_to_remove[batch_idx, self.eos_token_id] = False
                    else:
                        indices_to_remove[batch_idx, 784] = False



            elif prediction[last_s_idx:last_e_idx + 1] == "[var]":
                pred_last_idx = prediction.rindex(']')
                pred_var = prediction[pred_last_idx + 1:]

                for top_tok, top_tok_id in zip(top_tokens, top_token_ids):
                    pred_var_tmp = pred_var+top_tok
                    if len(pred_var) == 6: # [var] var_1 [
                        indices_to_remove[batch_idx, :] = True
                        if not check_match_flag:
                            indices_to_remove[batch_idx, 784] = False
                        indices_to_remove[batch_idx, self.eos_token_id] = False
                    elif len(pred_var) <= 5: # [var] var_1
                        if len(pred_var_tmp) > 6:
                            indices_to_remove[batch_idx, top_tok_id] = True
                        else:
                            indices_to_remove[batch_idx, top_tok_id] = False


            else:  # 排除这个预测 [...] 直接停止预测
                indices_to_remove[batch_idx, :] = True
                indices_to_remove[batch_idx, self.eos_token_id] = False
                # print(f"{idx0} no....")
        else:
            # print("structure constrained")

            pred_last_idx = prediction.rindex('[')
            pred_count = prediction.count('[')

            count = 0
            curr_s_idx = 0
            curr_e_idx = 0
            for idx, token in enumerate(structure_tokens):
                if token == "▁[":
                    count += 1
                    if count == pred_count:
                        curr_s_idx = idx

                if token == "]":
                    curr_e_idx = idx
                    if count == pred_count:
                        break

            pred_ph = prediction[pred_last_idx:]
            struct_ph_token_ids = structure_token_ids[curr_s_idx:curr_e_idx]
            struct_ph_tokens = structure_tokens[curr_s_idx:curr_e_idx]

            next_token_idx = -1
            for idx, spt in enumerate(struct_ph_token_ids):
                spt_decoded = self.tokenizer.batch_decode(structure_token_ids[curr_s_idx:curr_s_idx + idx + 1],
                                                          skip_special_tokens=True)
                spt_merged = "".join(spt_decoded)
                if spt_merged in pred_ph and len(spt_merged) <= len(pred_ph):
                    continue
                next_token_idx = idx
                break

            if next_token_idx > 0:
                ph_indices = self.tokenizer.convert_tokens_to_ids(['ent', 're', 'l', 'val', 'con', 'var'])
                indices_to_remove[batch_idx, ph_indices] = False

                next_token_id = struct_ph_token_ids[next_token_idx]
                indices_to_remove[batch_idx, :] = True
                indices_to_remove[batch_idx, next_token_id] = False
            else:
                indices_to_remove[batch_idx, :] = True
                indices_to_remove[batch_idx, 908] = False

