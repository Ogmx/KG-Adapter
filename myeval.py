import collections
import itertools
import lm_eval.metrics
import lm_eval.models
import lm_eval.tasks
import lm_eval.base
from lm_eval.base import BaseLM
import math, random
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import datasets

from tqdm import tqdm
import torch.nn.functional as F
from lm_eval import utils

from typing import List, Mapping, NewType, Optional, Tuple, Union
import torch
import transformers
from transformers import BatchEncoding

from model.llama import LlamaKgAdapterForCausalLM
from model.mistral_v1 import MistralKgAdapterForCausalLM

TokenSequence = Union[List[int], torch.LongTensor, torch.Tensor, BatchEncoding]
_DeviceMapping = NewType("DeviceMapping", Mapping[str, Union[int, str, torch.device]])


class MultiTokenEOSCriteria(transformers.StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence."""

    def __init__(
            self,
            sequence: str,
            tokenizer: transformers.PreTrainedTokenizer,
            initial_decoder_input_length: int,
            batch_size: int,
    ):
        self.initial_decoder_input_length = initial_decoder_input_length
        self.done_tracker = [False] * batch_size
        self.sequence = sequence
        self.sequence_ids = tokenizer.encode(sequence, add_special_tokens=False)
        self.sequence_id_len = len(self.sequence_ids)
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # For efficiency, we compare the last n tokens where n is the number of tokens in the stop_sequence
        lookback_ids_batch = input_ids[:, self.initial_decoder_input_length:][
                             :, -self.sequence_id_len:
                             ]

        lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)

        for i, done in enumerate(self.done_tracker):
            if not done:
                self.done_tracker[i] = self.sequence in lookback_tokens_batch[i]
        return False not in self.done_tracker


def stop_sequences_criteria(
        tokenizer: transformers.PreTrainedTokenizer,
        stop_sequences: List[str],
        initial_decoder_input_length: int,
        batch_size: int,
) -> transformers.StoppingCriteriaList:
    return transformers.StoppingCriteriaList(
        [
            *[
                MultiTokenEOSCriteria(
                    sequence, tokenizer, initial_decoder_input_length, batch_size
                )
                for sequence in stop_sequences
            ],
        ]
    )


class KGLLM(BaseLM):  # LLM+kg-adapter
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
            self,
            args,
            device="cuda",
            pretrained="gpt2",
            revision="main",
            low_cpu_mem_usage=None,
            subfolder=None,
            tokenizer=None,
            batch_size=1,
            max_batch_size=512,
            max_gen_toks=256,
            max_length=None,
            add_special_tokens: Optional[bool] = True,  # we manually add bos and eos token in text
            load_in_8bit: Optional[bool] = False,
            trust_remote_code: Optional[bool] = False,
            dtype: Optional[Union[str, torch.dtype]] = "auto",
    ):
        super().__init__()

        # Initialize model
        self.model = pretrained
        self._device = self.model.device
        if tokenizer:
            assert isinstance(
                tokenizer,
                transformers.PreTrainedTokenizer
            ) or isinstance(
                tokenizer,
                transformers.PreTrainedTokenizerFast
            )
            self.tokenizer = tokenizer
        else:
            # Get tokenizer
            model_name = self.model.name_or_path
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_name,
                revision=revision,
                trust_remote_code=trust_remote_code,
            )

        self.model.eval()

        self.vocab_size = self.tokenizer.vocab_size
        self._add_special_tokens = add_special_tokens
        self._config = args.model_config

        # Validate batch_size
        assert isinstance(batch_size, (int, str))

        # setup for automatic batch size detection
        if str(batch_size).startswith("auto"):
            batch_size = batch_size.split(":")
            self._batch_size = batch_size[0]
            self.batch_schedule = float(batch_size[1]) if len(batch_size) > 1 else 1
        else:
            self._batch_size = int(batch_size)
        self.max_batch_size = max_batch_size

        self._max_length = max_length
        self._max_gen_toks = max_gen_toks

    # copy from HuggingFaceAutoLM
    @property
    def eot_token(self) -> str:
        return self.tokenizer.eos_token

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def bot_token(self) -> str:
        return self.tokenizer.bos_token

    @property
    def bot_token_id(self) -> int:
        return self.tokenizer.bos_token_id

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    @property
    def max_length(self) -> int:
        """Return the maximum sequence length of the model.
        NOTE: Different model configurations have different max sequence length
        attribute names.
            - n_positions: (CTRLConfig, T5Config)
            - max_position_embeddings: (BartConfig, RoFormerConfig)
            - n_ctx: (GPT2Config)
        NOTE: For relative position encoded models you should specify the max
        sequence length of the model in the constructor via `max_length`.
        """
        if self._max_length is not None:
            return self._max_length
        # Try to get the sequence length from the model config.
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self._config, attr):
                return getattr(self._config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def batch_size(self) -> int:
        # TODO: Add adaptive batch size.
        return self._batch_size  # * gpus

    @property
    def device(self) -> Union[int, str, torch.device]:
        return self._device

    def tok_encode(self, string: str) -> TokenSequence:
        # TODO: Merge `tok_encode_batch` here.
        return self.tokenizer.encode(string, add_special_tokens=False)  # we manually add bos and eos token in text

    def tok_encode_batch(self, strings: List[str]) -> TokenSequence:
        return self.tokenizer(
            strings,
            padding=True,
            add_special_tokens=False,
            return_tensors="pt",
        )

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        whole_enc = self.tok_encode(context + continuation)
        context_enc = self.tok_encode(context)
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        return context_enc, continuation_enc

    def tok_decode(self, tokens: torch.LongTensor) -> List[str]:
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def greedy_until(
            self, requests: List[Tuple[str, Union[List[str], str]]]
    ) -> List[str]:
        def _collate(x):
            tokens = self.tok_encode(x[0])
            return len(tokens), x[0]

        results = []
        reorder = utils.Reorderer(requests, _collate)

        adaptive_batch_size = None
        if self.batch_size == "auto":
            # using rolling window with maximum context
            print("Passed argument batch_size = auto. Detecting largest batch size")
            batch_size = self._detect_batch_size()
            print(f"Determined Largest batch size: {batch_size}")
            adaptive_batch_size = batch_size

        for chunk in utils.chunks(
                tqdm(reorder.get_reordered(), disable=False),
                self.batch_size if self.batch_size != "auto" else adaptive_batch_size,
        ):
            context = [c[0] for c in chunk]
            request_args = chunk[0][1]
            stop = request_args.get("until", None)
            stop_sequences = stop if isinstance(stop, list) else [stop]
            max_generation_length = request_args.get("max_length", None)

            assert (
                    isinstance(max_generation_length, int) or max_generation_length is None
            )
            assert isinstance(stop_sequences, list) or stop_sequences is None

            # TODO: Find a better way to handle stop sequences for 0-shot.
            if stop_sequences is None:
                until = [self.eot_token]
            else:
                until = stop_sequences + [self.eot_token]

            if max_generation_length is None:
                max_tokens = self.max_gen_toks
            else:
                max_tokens = max_generation_length

            token_context = self.tok_encode_batch(context)

            responses = self._model_generate(
                inputs=token_context,
                max_tokens=max_tokens,
                stop=until,
            )
            responses = self.tok_decode(responses.tolist())

            for response in responses:
                # Ensure the generated responses do not contain the stop sequences.
                for term in until:
                    response = response.split(term)[0]
                # partial caching
                self.cache_hook.add_partial("greedy_until", (context, until), response)
                results.append(response)
        return reorder.get_original(results)

    def _model_call(self, inputs, mask, sg, labels=None):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        if sg is not None:
            with torch.no_grad():   # for our kg-adapter model
                return self.model(input_ids=inputs, labels=labels, attention_mask=mask, sg=sg)
        else:
            with torch.no_grad():   # for baseline models
                return self.model(input_ids=inputs, attention_mask=mask)

    def _model_generate(
            self,
            inputs: transformers.BatchEncoding,
            max_tokens: int,
            stop: Optional[List[str]] = None,
    ) -> TokenSequence:
        # Ensure that the context does not encroach into the `space`
        # for the generation.
        input_ids = inputs["input_ids"][:, self.max_gen_toks - self.max_length:]
        attention_mask = inputs["attention_mask"][
                         :, self.max_gen_toks - self.max_length:
                         ]
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, input_ids.shape[1], input_ids.shape[0]
        )

        generations = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # GPT style models require the `generate` `max_length` arg to include the
            # context length, so we instead set `max_new_tokens` which is the number
            # of new tokens to generate, excluding the current number of tokens.
            max_new_tokens=max_tokens,
            stopping_criteria=stopping_criteria,
            do_sample=False,
        )
        return utils.select_continuation_from_batch_left_padding(
            generations, max_context_size=inputs["input_ids"].size(1)
        )

    # def loglikelihood(self, requests):
    #     new_reqs = []
    #     args = requests[-1]
    #     requests = requests[:-1]
    #     if len(requests[0]) > 2:
    #         for i in range(len(requests)):
    #             context = requests[i][0][0]
    #             continuation = requests[i][0][1]
    #             pre_data = requests[i][2]
    #             sg = pre_data['sg']
    #
    #             # if context == "":
    #             #     # end of text as context
    #             #     context_enc, continuation_enc = [self.eot_token_id], self.tok_encode(
    #             #         continuation
    #             #     )
    #             # else:
    #             #     context_enc, continuation_enc = self._encode_pair(context, continuation)
    #             context_enc = pre_data['input_ids_no_response'].tolist()
    #             continuation_enc = pre_data['labels'][pre_data['labels'] != -100].tolist()
    #             # add: not use special tokens
    #             if context_enc[0] == 1:
    #                 context_enc = context_enc[1:]
    #             if continuation_enc[-1] == 2:
    #                 continuation_enc = continuation_enc[:-1]
    #
    #             new_reqs.append(((context, continuation), context_enc, continuation_enc, sg))
    #     else:
    #         from mydata import kg_adapter_left_pad_collate_fn
    #         for chunk in utils.chunks(requests, self.batch_size):
    #             input_lst, sg_lst = zip(*chunk)
    #             context_lst = []
    #             continuation_lst = []
    #
    #             for x in input_lst:
    #                 context = x[0]
    #                 if len(context) == 0:                               # Fill empty context with bos token
    #                     context_lst.append(f"{self.bot_token}")
    #                 elif context.startswith(f"{self.bot_token}"):       # If context not startswith bos add it or not
    #                     context_lst.append(context)
    #                 else:
    #                     context_lst.append(f"{self.bot_token}{context}")
    #
    #                 continuation = x[1].strip()
    #                 if len(continuation) == 0:                          # Fill empty label with eos token
    #                     continuation_lst.append(f"{self.eot_token}")
    #                 elif context.endswith(f"{self.eot_token}"):         # If label not endswith eos add it or not
    #                     continuation_lst.append(continuation)
    #                 else:
    #                     continuation_lst.append(f"{continuation}{self.eot_token}")
    #
    #             data = []
    #             for i, (inputs, labels) in enumerate(zip(context_lst, continuation_lst)):
    #                 idx = torch.tensor(i).cpu()
    #                 label_ids = self.tok_encode(labels)
    #                 input_no_res_ids = self.tok_encode(inputs)
    #                 prompt_len = torch.tensor(len(input_no_res_ids)).squeeze().cpu()
    #                 labels = torch.tensor(input_no_res_ids + label_ids).cpu()
    #                 input_ids = torch.tensor(input_no_res_ids + label_ids).cpu()
    #                 labels[:prompt_len] = -100
    #                 assert labels.size(0) <= self.max_length
    #                 sg = sg_lst[i].clone().detach().cpu()
    #                 data.append([idx, input_ids, labels, prompt_len, sg])
    #
    #             # use the same data collate function as train and validation
    #             idx, x, y, mask, prompt_len, x_no_res, x_no_res_mask, sg = kg_adapter_left_pad_collate_fn(data, args)
    #             processed_data = (x, y, mask, prompt_len, x_no_res, x_no_res_mask, sg)
    #
    #             new_reqs.append(((context_lst, continuation_lst), processed_data))
    #
    #     return self._loglikelihood_tokens(new_reqs, args)

    # def _loglikelihood_tokens(
    #     self,
    #     requests, args=None,
    #     disable_tqdm: Optional[bool] = False,
    # ) -> List[Tuple[float, bool]]:
    #     results = []
    #     for chunk in tqdm(
    #         requests, total=math.ceil(len(requests)), disable=disable_tqdm
    #     ):
    #         cache_keys, processed_data = chunk
    #         x, y, mask, prompt_len, x_no_res, x_no_res_mask, sg = processed_data
    #         inputs_tokens = x.to(self.device)
    #         targets_tokens = y.to(self.device)
    #         mask = mask.to(self.device)
    #         sg = sg.to(self.device)
    #
    #         outputs = self._model_call(inputs=inputs_tokens, labels=targets_tokens,
    #                                    mask=mask, sg=sg)
    #
    #         log_softmaxes = F.log_softmax(outputs.logits, dim=-1)
    #
    #         output_iterator = zip(
    #             zip(cache_keys[0], cache_keys[1]),
    #             log_softmaxes,
    #             targets_tokens,
    #             prompt_len,
    #         )
    #         for cache_key, log_softmax, target_tokens, _ in output_iterator:
    #             length = (target_tokens != -100).sum().item()
    #             log_softmax = log_softmax[-length:]
    #             target_tokens = target_tokens[-length:]
    #             greedy_tokens = log_softmax.argmax(dim=-1)
    #             max_equal = (greedy_tokens == target_tokens).all()
    #             target_logits = torch.gather(
    #                 log_softmax, 1, target_tokens.unsqueeze(-1)
    #             ).squeeze(-1)
    #             answer = (float(target_logits.sum()), bool(max_equal))
    #             results.append(answer)
    #             if cache_key is not None:
    #                 self.cache_hook.add_partial("loglikelihood", cache_key, answer)
    #     return results

    def loglikelihood(self, requests):
        new_reqs = []
        args = requests[-1]
        requests = requests[:-1]
        for (context, continuation), sg in requests:
            if self._add_special_tokens:
                if len(context) == 0:  # Fill empty context with bos token
                    context = f"{self.bot_token}"
                elif context.startswith(f"{self.bot_token}"):  # If context not startswith bos add it or not
                    context = context
                else:
                    context = f"{self.bot_token}{context}"

                continuation = continuation.strip()
                if len(continuation) == 0:  # Fill empty label with eos token
                    continuation = f"{self.eot_token}"
                elif context.endswith(f"{self.eot_token}"):  # If label not endswith eos add it or not
                    continuation = continuation
                else:
                    continuation = f"{continuation}{self.eot_token}"

            if context == "":
                # end of text as context
                context_enc, continuation_enc = [self.eot_token_id], self.tok_encode(
                    continuation
                )
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc, sg))

        return self._loglikelihood_tokens(new_reqs, args)

    def _loglikelihood_tokens(self, requests, args=None, disable_tqdm=False, override_bs=None):
        # TODO: implement some kind of efficient-request-middleware that lumps together requests with the same context
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        re_ord = utils.Reorderer(requests, _collate)

        reordered_requests = re_ord.get_reordered()
        n_reordered_requests = len(reordered_requests)

        # automatic (variable) batch size detection for vectorization
        # pull longest context sample from request
        def _batch_scheduler(pos):
            sched = pos // int(n_reordered_requests / self.batch_schedule)
            if sched in self.batch_sizes:
                return self.batch_sizes[sched]
            print(
                f"Passed argument batch_size = auto:{self.batch_schedule}. Detecting largest batch size"
            )
            self.batch_sizes[sched] = self._detect_batch_size(reordered_requests, pos)
            print(f"Determined largest batch size: {self.batch_sizes[sched]}")
            return self.batch_sizes[sched]

        for chunk in utils.chunks(
                tqdm(reordered_requests, disable=disable_tqdm),
                n=self.batch_size
                if self.batch_size != "auto"
                else override_bs
                if override_bs is not None
                else 0,
                fn=_batch_scheduler
                if self.batch_size == "auto" and n_reordered_requests > 0 and not override_bs
                else None,
        ):
            inps = []
            cont_toks_list = []
            inplens = []
            padding_length = None
            sg_lst = []
            # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
            # tensors, then we pack them together into a batch, call the model, and then pick it all apart
            # again because vectorizing is annoying

            for _, context_enc, continuation_enc, sg in chunk:
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # how this all works:
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # gpt2    \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice
                if sg is not None:
                    sg_lst.append(sg.clone())
                else:
                    sg_lst.append(None)
                # when too long to fit in context, truncate from the left
                inp = torch.tensor(
                    (context_enc + continuation_enc)[-(self.max_length + 1):][:-1],
                    dtype=torch.long,
                ).to(self.device)
                (inplen,) = inp.shape

                cont = continuation_enc

                # since in _collate we make sure length is descending, the longest is always the first one.
                padding_length = (
                    padding_length if padding_length is not None else inplen
                    # + 1        # change: +1 to support align_mask
                )

                # pad length from seq to padding_length
                inp = torch.cat(
                    [
                        inp,  # [seq]
                        # add: not all model use 0 as pad_id
                        torch.full((padding_length - inplen,), args.pad_id, dtype=torch.long).to(
                            inp.device),
                        # torch.zeros(padding_length - inplen, dtype=torch.long).to(
                        #     inp.device
                        # ),  # [padding_length - seq]
                    ],
                    dim=0,
                )

                inps.append(inp.unsqueeze(0))  # [1, padding_length]
                cont_toks_list.append(cont)
                inplens.append(inplen)

            # build batch sg data
            from mydata import kg_adapter_right_pad_collate_fn

            tmp_data = []
            for i in range(len(sg_lst)):
                sg_tmp = sg_lst[i].clone().detach().cpu() if sg_lst[i] is not None else None
                tmp_data.append([torch.tensor(i).cpu(), inps[i].clone().detach().squeeze().cpu(),
                                 torch.tensor(cont_toks_list[i]).cpu(), torch.tensor(inplens[i]).cpu(),
                                 sg_tmp])
            _, _, _, _, _, _, _, sg = kg_adapter_right_pad_collate_fn(tmp_data, args)
            sg.to(self.device)
            if 'peft' in args.peft_type or 'base' in args.peft_type:
                sg = None

            batched_inps = torch.cat(inps, dim=0)  # [batch, padding_length]
            mask = batched_inps != 0
            output = self._model_call(inputs=batched_inps,
                                      mask=mask,
                                      sg=sg).logits
            multi_logits = F.log_softmax(output, dim=-1).cpu()  # [batch, padding_length, vocab]

            for (cache_key, _, _, _), logits, inp, inplen, cont_toks in zip(
                    chunk, multi_logits, inps, inplens, cont_toks_list
            ):

                # Slice to original seq length
                contlen = len(cont_toks)
                inplen = inplen + (logits.shape[
                                       0] - padding_length)  # if "virtual tokens" (from prompt tuning) are added, inplen is larger
                logits = logits[inplen - contlen: inplen].unsqueeze(0)  # [1, seq, vocab]

                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(dim=-1)
                cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(0)  # [1, seq]
                max_equal = (greedy_tokens == cont_toks).all()

                # Obtain log-probs at the corresponding continuation token indices
                # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)  # [1, seq]

                # Answer: (log prob, is-exact-match)
                answer = (float(logits.sum()), bool(max_equal))

                # partial caching
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)

                res.append(answer)

        return re_ord.get_original(res)


decontaminate_suffix = "_decontaminate"


def evaluate(
        lm,
        task_dict,
        prompt,
        user_instruction,
        system_instruction,
        add_special_tokens,
        kg=None,
        use_kg=True,
        replace=True,
        args=None,
        preprocessed_data=None,
        provide_description=None,
        num_fewshot=0,
        limit=None,
        bootstrap_iters=100000,
        description_dict=None,
        decontamination_ngrams_path=None,
        write_out=False,
        output_base_path=None,
        epoch="test",
):
    """Instantiate and evaluate a model on a list of tasks.

    :param lm: obj
        Language Model
    :param task_dict: dict[str, Task]
        Dictionary of tasks. Tasks will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param provide_description: bool
        Not implemented, and this option is deprecated and will be removed in a future version in favor of a different description providing method
    :param num_fewshot: int
        Number of examples in few-shot context
    :param limit: int, optional
        Limit the number of examples per task (only use this for testing)
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param description_dict: dict[str, str]
        Dictionary of custom task descriptions of the form: `task_name: description`
    :param write_out: bool
        If True, write all prompts, logits and metrics to json for offline analysis
    :param output_base_path: str, optional
        Directory to which detailed eval info will be written. Defaults to present working dir
    :return
        Dictionary of results
    """
    assert not provide_description  # not implemented.
    if provide_description is not None:
        # nudge people to not specify it at all
        print(
            "WARNING: provide_description is deprecated and will be removed in a future version in favor of description_dict"
        )

    decontaminate = decontamination_ngrams_path is not None

    task_dict_items = [
        (name, task)
        for name, task in task_dict.items()
        if (task.has_validation_docs() or task.has_test_docs())
    ]

    results = collections.defaultdict(dict)
    versions = collections.defaultdict(dict)

    requests = collections.defaultdict(list)
    requests_origin = collections.defaultdict(list)

    overlaps = collections.defaultdict(list)  # {task_name: contaminated_docs}

    # If we ever run into issues where the eval tasks don't fit in memory and we can't afford a machine with bigger
    # memory, we can always modify this plumbing to support that, but I didn't want to include it just yet because
    # over-engineering is bad (or we could make it write the requests to disk and then read them back out again
    #  - probably using an sqlite db because of all the moving parts we have

    docs = {}
    write_out_info = {}

    docs_for_decontamination = collections.defaultdict(list)

    # get lists of each type of request
    for task_name, task in task_dict_items:
        versions[task_name] = task.VERSION
        # default to test doc, fall back to val doc if validation unavailable
        if task.has_test_docs():
            task_doc_func = task.test_docs
            task_set = "test"  # Required for caching in the decontamination
        elif task.has_validation_docs():
            task_set = "val"  # Required for caching in the decontamination
            task_doc_func = task.validation_docs
        else:
            raise RuntimeError("Task has neither test_docs nor validation_docs")

        # deterministically shuffle docs and chop off the first `limit` because sometimes docs are in some kind of order
        task_docs = list(task_doc_func())
        rnd = random.Random()
        # rnd.seed(42)
        # rnd.shuffle(task_docs)
        print(f"Task: {task_name}; number of docs: {len(task_docs)}")

        if write_out:
            prompt_details = []

        description = (
            description_dict[task_name]
            if description_dict and task_name in description_dict
            else ""
        )
        if limit is not None:
            limit = int(len(task_docs) * limit) if limit < 1.0 else int(limit)

        if not args.debug and kg is not None:
            assert len(kg) == len(task_docs)

        for doc_id, doc in enumerate(itertools.islice(task_docs, 0, limit)):
            if decontaminate and task.should_decontaminate():
                docs_for_decontamination[(task_name, task_set)].append(
                    task.doc_to_decontamination_query(doc)
                )
            docs[(task_name, doc_id)] = doc
            if "kg" in task_name:  # my task
                ctx = task.fewshot_context(
                    doc=doc, num_fewshot=num_fewshot, rnd=rnd, description=description,
                    add_special_token=add_special_tokens,
                    prompt=prompt,
                    user_instruction=user_instruction,
                    system_instruction=system_instruction,
                    replace=replace,
                )
            else:
                ctx = task.fewshot_context(
                    doc=doc, num_fewshot=num_fewshot, rnd=rnd, description=description,

                )
            reqs = task.construct_requests(doc, ctx)

            if use_kg and kg is None:
                sg = task.get_sg(doc_id)
            elif kg is not None:
                sg = kg[doc_id]
            else:
                sg = None

            if write_out:
                prompt_details.append({"doc_id": doc_id})

            # print the prompt for the first few documents
            if doc_id < 1 and epoch == 'test':
                print(
                    f"Task: {task_name}; document {doc_id}; context prompt (starting on next line):\n{ctx}\n(end of prompt on previous line)"
                )
                print("Requests:", reqs)

            if not isinstance(reqs, (list, tuple)):
                reqs = [reqs]
            for i, req in enumerate(reqs):
                # add: insert sg data
                requests[req.request_type].append([req, sg])
                # i: index in requests for a single task instance
                # doc_id: unique id that we can get back to a doc using `docs`
                requests_origin[req.request_type].append((i, task_name, doc, doc_id))

                if write_out:
                    prompt_details[-1][f"prompt_{i}"] = "".join(
                        (map(lambda x: "".join(x), req.args))
                    )

        if write_out:
            write_out_info[task_name] = prompt_details

    # Compare all tasks/sets at once to ensure a single training set scan
    if decontaminate:
        from lm_eval.decontamination.decontaminate import get_train_overlap

        print("Finding train/test overlap, please wait...")
        overlaps = get_train_overlap(
            docs_for_decontamination, decontamination_ngrams_path, limit
        )

    # all responses for each (task, doc)
    process_res_queue = collections.defaultdict(list)

    # execute each type of request
    for reqtype, reqs in requests.items():
        print("Running", reqtype, "requests")
        # add: we preprocessed some data because we use kg data that not support by harness
        if preprocessed_data is not None:
            assert len(preprocessed_data) == len(reqs)
            inp = []
            for i in range(len(reqs)):
                inp.append((reqs[i][0].args, reqs[i][1], preprocessed_data[i]))
            inp.append(args)
            resps = getattr(lm, reqtype)(inp)
        else:
            inp = []
            for i in range(len(reqs)):
                inp.append((reqs[i][0].args, reqs[i][1]))
            inp.append(args)
            resps = getattr(lm, reqtype)(inp)

        resps = [
            x if req[0].index is None else x[req[0].index] for x, req in zip(resps, reqs)
        ]

        for resp, (i, task_name, doc, doc_id) in zip(resps, requests_origin[reqtype]):
            process_res_queue[(task_name, doc_id)].append((i, resp))

            if write_out:
                write_out_info[task_name][doc_id][f"logit_{i}"] = resp
                task = task_dict[task_name]
                if isinstance(task, lm_eval.base.MultipleChoiceTask):
                    write_out_info[task_name][doc_id]["truth"] = doc["gold"]
                elif isinstance(task, lm_eval.tasks.winogrande.Winogrande):
                    write_out_info[task_name][doc_id]["truth"] = task.answer_to_num[
                        doc["answer"]
                    ]
                else:
                    write_out_info[task_name][doc_id]["truth"] = task.doc_to_target(doc)

    vals = collections.defaultdict(list)

    # unpack results and sort back in order and return control to Task
    for (task_name, doc_id), requests in process_res_queue.items():
        requests.sort(key=lambda x: x[0])
        requests = [x[1] for x in requests]

        task = task_dict[task_name]
        doc = docs[(task_name, doc_id)]

        metrics = task.process_results(doc, requests)
        for metric, value in metrics.items():
            vals[(task_name, metric)].append(value)

            if write_out:
                write_out_info[task_name][doc_id][metric] = str(value)

            # Re-use the evaluation for the decontaminated set by just ignoring the overlaps
            if decontaminate and task_name in overlaps:
                if doc_id not in overlaps[task_name]:
                    vals[(task_name, metric + decontaminate_suffix)].append(value)

    # aggregate results
    for (task_name, metric), items in vals.items():
        task = task_dict[task_name]
        real_metric = metric  # key when looking up the metric with task.aggregation
        if metric.endswith(decontaminate_suffix):
            real_metric = metric.replace(
                decontaminate_suffix, ""
            )  # decontaminated still uses the same metric
        results[task_name][metric] = task.aggregation()[real_metric](items)

        # hotfix: bleu, chrf, ter seem to be really expensive to bootstrap
        # so we run them less iterations. still looking for a cleaner way to do this

        stderr = lm_eval.metrics.stderr_for_metric(
            metric=task.aggregation()[real_metric],
            bootstrap_iters=min(bootstrap_iters, 1000)
            if metric in ["bleu", "chrf", "ter"]
            else bootstrap_iters,
        )

        if stderr is not None:
            results[task_name][metric + "_stderr"] = stderr(items)

    if write_out:
        import json
        import pathlib

        output_base_path = (
            pathlib.Path(output_base_path)
            if output_base_path is not None
            else pathlib.Path(".")
        )
        try:
            output_base_path.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            pass

        for task_name, _ in task_dict_items:
            with open(
                    output_base_path.joinpath(f"{task_name}_ep_{epoch}_write_out_info.json"),
                    "w",
                    encoding="utf8",
            ) as fp:
                json.dump(write_out_info[task_name], fp, indent=4, ensure_ascii=False)

    return {"results": dict(results), "versions": dict(versions)}


def llm_eval(model, args=None, tokenizer=None, epoch="test"):
    from lm_eval.tasks import truthfulqa
    from eval import truthfulqa_kg, halueval_kg, obqa_kg, csqa_kg, webqsp_kg, cwq_kg
    import lightning as L
    L.seed_everything(42, workers=True)
    eval_data_lst = ["tuqa", "tuqa_kg", "halu_kg", "csqa_kg", "obqa_kg", "cwq_kg", "wqsp_kg"]
    # build task
    truthfulqa_kg.TruthfulQAMultipleChoice.DATASET_PATH = "/raid_sdb/home/tsy/datasets/huggingface_datasets/truthful_qa"
    truthfulqa.TruthfulQAMultipleChoice.DATASET_PATH = "/raid_sdb/home/tsy/datasets/huggingface_datasets/truthful_qa"
    halueval_kg.HaluEval.DATASET_PATH = "/raid_sdb/home/tsy/datasets/huggingface_datasets/halu_eval"
    obqa_kg.OpenBookQA.DATASET_PATH = "/raid_sdb/home/tsy/datasets/huggingface_datasets/openbookqa"
    csqa_kg.CSQA.DATASET_PATH = "/raid_sdb/home/tsy/datasets/huggingface_datasets/commonsense_qa"
    webqsp_kg.WebQSP.DATASET_PATH = "/raid_sdb/home/tsy/datasets/huggingface_datasets/webqsp"
    cwq_kg.CWQ.DATASET_PATH = "/raid_sdb/home/tsy/datasets/huggingface_datasets/cwq"

    # load kg data
    preprocessed_data = None
    test_data = None
    sg = None

    if "use_predata" in args.test_set:
        print(f"using preprocessed harness data")
        preprocessed_data = torch.load(
            f"/raid_sdb/home/tsy/KG_data/preprocessed_harness_data_{args.eval_data_version}.pt")
    else:
        print(f"using kg data: {args.data_path}/test_{args.test_data_version}.pt")
        test_data = torch.load(f"{args.data_path}/test_{args.test_data_version}.pt")
        if "no_kg" in args.test_set:
            sg = None
        elif "tuqa" in args.test_set:
            sg = [x['sg'] for x in test_data[817:817 * 2]]  # use mc2 task sg
        else:
            sg = [x['sg'] for x in test_data]

    if "no_kg" in args.test_set:
        sg = None

    task_dict = {
        'tuqa_kg': truthfulqa_kg.TruthfulQAMultipleChoice(
            download_mode=datasets.DownloadMode.REUSE_DATASET_IF_EXISTS),
        'tuqa': truthfulqa.TruthfulQAMultipleChoice(
            download_mode=datasets.DownloadMode.REUSE_DATASET_IF_EXISTS),
        'halu_kg': halueval_kg.HaluEval(download_mode=datasets.DownloadMode.REUSE_DATASET_IF_EXISTS),
        'csqa_kg': csqa_kg.CSQA(download_mode=datasets.DownloadMode.REUSE_DATASET_IF_EXISTS),
        'obqa_kg': obqa_kg.OpenBookQA(download_mode=datasets.DownloadMode.REUSE_DATASET_IF_EXISTS),
        'cwq_kg': cwq_kg.CWQ(download_mode=datasets.DownloadMode.REUSE_DATASET_IF_EXISTS),
        'wqsp_kg': webqsp_kg.WebQSP(download_mode=datasets.DownloadMode.REUSE_DATASET_IF_EXISTS),
    }
    for name in eval_data_lst:
        if name not in args.test_set:
            del task_dict[name]
        else:
            # TODO: finish here
            if isinstance(test_data, dict):
                sg = [x['sg'] for x in test_data[name]]
                task_dict[name].set_sg(sg)

    if len(task_dict) == 0:
        print("not use harness eval")
        return None

    batch_size = 8 if args is None else args.micro_batch_size * 4
    output_base_path = None if args is None else args.out_dir + args.exp_name + "/results"

    # build llm-eval model
    lm = KGLLM(
        args=args,
        pretrained=model,
        tokenizer=tokenizer if tokenizer is not None else model.tokenizer,
        batch_size=batch_size,
        max_batch_size=16,
        add_special_tokens=True if "add_special_tokens" in args.test_set else False
    )

    # get prompt type by model
    if "llama" in args.pretrained_path.lower():
        prompt = "llama-chat"
    elif "mistral" in args.pretrained_path.lower():
        prompt = "mistral"
    elif "openorca" in args.pretrained_path.lower() or "orca" in args.pretrained_path.lower():
        prompt = "orca"
    elif "zephyr" in args.pretrained_path.lower():
        prompt = "zephyr"
    else:
        prompt = "default"

    results = evaluate(
        lm=lm,
        prompt=prompt,  # default, llama-chat, mistral, orca, zephyr
        user_instruction=None if "no_user_inst" in args.test_set else "my",
        system_instruction="task" if "task_system_inst" in args.test_set else None,
        add_special_tokens=True if "add_special_tokens" in args.test_set else False,
        replace=False if "not_replace" in args.test_set else True,
        use_kg=False if "no_kg" in args.test_set else True,
        args=args,
        kg=sg,
        preprocessed_data=preprocessed_data,
        task_dict=task_dict,
        limit=100 if args.debug else None,
        num_fewshot=0,
        write_out=True,
        output_base_path=output_base_path,
        epoch=epoch
    )
    dumped = json.dumps(results, indent=2)
    print(dumped)

    return results
