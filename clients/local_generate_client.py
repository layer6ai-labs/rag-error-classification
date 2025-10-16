import logging

import numpy as np
import numpy.typing as npt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from baseconfig import GeneratorConfig
from tqdm import trange


class LocalGenerateClient:
    def __init__(
        self,
        config: GeneratorConfig,
    ) -> None:
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.info("Initiating Generate Client...")
        self._tokenizer_path = config.tokenizer
        self._context_size = config.context_size
        self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_path, padding_side="left")
        self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        self._num_output_token_cache = {}

        self.SYS_PROMPT = (
            "You are an assistant for answering queries."
            "You are given the a list of context (extracted parts of some documents) and a query."
            "Base on the given context, provide an answer to the query."
            "Please be concise and to the point."
            "If you don't know the answer say 'I don't know!' Don't make up an answer."
            "Cite the document id used. The output format be answer with citations. "
            "Only respond with the answer, do not explain."
        )
        self.config = config
        self._local_llm = AutoModelForCausalLM.from_pretrained(
            config.model, device_map=config.device
        )
        self.log.info("Done.")

    def run_llm(
        self,
        prompt: list[dict[str, str]],
        return_logits: bool = False,
        device: str | torch.device | None = None,
    ) -> tuple[str, npt.NDArray | None]:
        device = device if device is not None else self.config.device
        input_ids = self._tokenizer.apply_chat_template(
            prompt, add_generation_prompt=True, return_tensors="pt"
        ).to(device)

        terminators = [
            self._tokenizer.eos_token_id,
            self._tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        output = self._local_llm.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            pad_token_id=self._tokenizer.eos_token_id,
            do_sample=False,
            temperature=None,
            top_p=None,
            output_logits=return_logits,
            return_dict_in_generate=True,
        )
        idx = input_ids.shape[-1]
        output_ids = output.sequences[0, idx:]
        if return_logits:
            stacked_logits = torch.stack(output.logits)[:, 0, :]
            stacked_logits = torch.log_softmax(stacked_logits, dim=1)
            output_logits = torch.take_along_dim(
                stacked_logits, output_ids[:, None], dim=1
            ).reshape(-1)
        else:
            output_logits = None
        outputs = self._tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )
        return outputs, output_logits

    def run_llm_batch(
        self,
        prompt: list[list[dict[str, str]]],
        return_logits: bool,
        device: str | torch.device | None = None,
    ) -> tuple[list[str], npt.NDArray]:
        device = device if device is not None else self.config.device
        formatted_prompt = self._tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )

        inputs = self._tokenizer(formatted_prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        terminators = [
            self._tokenizer.eos_token_id,
            self._tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        output_list = self._local_llm.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            pad_token_id=self._tokenizer.eos_token_id,
            attention_mask=attention_mask,
            do_sample=False,
            temperature=None,
            top_p=None,
            output_logits=return_logits,
            return_dict_in_generate=True,
        )
        idx = input_ids.shape[-1]
        output_ids = output_list.sequences[:, idx:]
        if return_logits:
            stacked_logits = torch.stack(output_list.logits)
            stacked_logits = torch.log_softmax(stacked_logits, dim=2)
            output_logits = torch.take_along_dim(
                stacked_logits, output_ids.transpose(0, 1)[:, :, None], dim=2
            )[:, :, 0].transpose(0, 1)
            # putting only nans for second eos_token_id onwards for each row.
            eos_mask = (output_ids == self._tokenizer.eos_token_id)[:, :-1]
            output_logits[:, 1:][eos_mask] = np.nan
            output_logits = output_logits.cpu().numpy()
        else:
            output_logits = np.empty(shape=(len(output_list), 0))
        outputs = self._tokenizer.batch_decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )
        return outputs, output_logits

    def generate(
        self,
        query: str,
        context_docs: npt.NDArray,
        return_logits: bool = False,
        device: str | torch.device | None = None,
    ) -> tuple[str, npt.NDArray | None]:
        """
        Generate answer according to the context
        """
        prompt = self.create_prompt(query, context_docs)
        answer = self.run_llm(prompt, return_logits, device=device)
        return answer

    def generate_batch(
        self,
        query: list[str],
        context_docs: list[npt.NDArray],
        batch_size: int = 1,
        return_logits: bool = False,
        device: str | torch.device | None = None,
    ) -> tuple[list[str], npt.NDArray]:
        """
        Generate answer according to the context
        """
        num_batches = -(-len(query) // batch_size)
        answer_list = []
        logit_list = []
        for batch_id in trange(num_batches, total=num_batches):
            start = batch_id * batch_size
            end = min(start + batch_size, len(query))
            prompt = [
                self.create_prompt(q, context_docs)
                for q, context_docs in zip(query[start:end], context_docs[start:end])
            ]
            text, logit = self.run_llm_batch(prompt, return_logits, device=device)
            print(text)
            answer_list.extend(text)
            logit_list.append(logit)

        # concatenate logits
        max_logit_length = max(logit.shape[1] for logit in logit_list)
        logit_list = [
            np.concatenate(
                [
                    logit,
                    np.full((len(logit), max_logit_length - logit.shape[1]), fill_value=np.nan),
                ],
                axis=1,
            )
            for logit in logit_list
        ]
        logit = np.concatenate(logit_list, axis=0)

        return answer_list, logit

    def create_prompt(
        self,
        query_str: str,
        context_docs: npt.NDArray,
    ) -> list[dict[str, str]]:
        context_str = ""
        # ('290', '290_0',
        # 'A\n\nA (named , plural "As", ...',
        # array([0.41795746]))
        for i, c in enumerate(context_docs):
            context_str += "{" + f"document id: {c['chunk_id']}, content: {c['text']}" + "}"

        user_prompt = (
            # "Context information is below.\n"
            # "---------------------\n"
            # "{context_str}\n"
            # "---------------------\n"
            # "Base on the given context, provide an answer to the query."
            # "If you don't know the answer say 'I don't know!' Don't make up an answer."
            # "Cite the document id used. The output format be answer with citations."
            # "For example: Answer: Paris. [1]"
            # "Only respond with the answer, do not explain."
            # f"Query:\n{query_str}\n"
            f"Query: \n{query_str}\n"
            f"Context:\n{context_str}\n"
        )
        prompt, num_tokens = self.generate_prompt_prefix()
        if self.get_num_tokens(user_prompt) + num_tokens > self._context_size:
            raise ValueError("Prompt exceeds context size")
        prompt.append({"role": "user", "content": user_prompt})
        return prompt

    def get_num_tokens(self, prompt: str) -> int:
        return len(self._tokenizer.encode(prompt))

    def generate_prompt_prefix(self) -> tuple[list[dict[str, str]], int]:
        example_user_prompts = [
            (
                "Query\n: Where is the capital of France? Context:\n"
                "{document id: 1, content: Paris is the capital of France, "
                "the largest country of Europe with 550 000 km2).} "
                "{document id: 2, content: France is a country in Europe.}]"
            )
        ]
        example_assistant_response = ["Answer: Paris. [1]"]

        prompt_prefix = [{"role": "system", "content": self.SYS_PROMPT}]
        for i in range(len(example_user_prompts)):
            prompt_prefix.append({"role": "user", "content": example_user_prompts[i]})
            prompt_prefix.append({"role": "assistant", "content": example_assistant_response[i]})
        num_tokens = (
            self.get_num_tokens(self.SYS_PROMPT)
            + self.get_num_tokens(example_user_prompts[0])
            + self.get_num_tokens(example_assistant_response[0])
        )
        return prompt_prefix, num_tokens
