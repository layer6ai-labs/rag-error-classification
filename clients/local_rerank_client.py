# Code in this file is borrowed from https://github.com/castorini/rank_llm

import logging
import re

import numpy as np
import numpy.typing as npt
import torch
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from ftfy import fix_text
from baseconfig import RerankConfig


class LocalRerankClient:
    def __init__(
        self,
        config: RerankConfig,
    ) -> None:
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.info("Initiating Rerank Client...")
        self._tokenizer_path = config.tokenizer
        self._context_size = config.context_size
        self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_path, padding_side="left")
        self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        self._num_output_token_cache = {}
        self._local_llm = AutoModelForCausalLM.from_pretrained(
            config.model, device_map=config.device
        )
        self.config = config
        self._local_llm.eval()
        self.log.info("Done.")

    def run_llm(
        self, prompt: str, current_window_size: int, device: str | torch.device | None = None
    ) -> str:
        device = device if device is not None else self.config.device
        inputs = self._tokenizer([prompt], return_tensors="pt").to(device)
        gen_cfg = GenerationConfig.from_model_config(self._local_llm.config)
        gen_cfg.max_new_tokens = self.num_output_tokens(current_window_size)
        gen_cfg.min_new_tokens = self.num_output_tokens(current_window_size)
        gen_cfg.do_sample = False
        with torch.no_grad():
            output_ids = self._local_llm.generate(**inputs, generation_config=gen_cfg)

        if self._local_llm.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(inputs["input_ids"][0]) :]  # noqa: E203
        outputs = self._tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )
        return outputs

    def run_llm_batch(
        self, prompts: list[str], current_window_size: int, device: str | torch.device | None = None
    ) -> list[str]:
        device = device if device is not None else self.config.device
        gen_cfg = GenerationConfig.from_model_config(self._local_llm.config)
        gen_cfg.max_new_tokens = self.num_output_tokens(current_window_size)
        gen_cfg.min_new_tokens = self.num_output_tokens(current_window_size)
        gen_cfg.do_sample = False
        inputs = self._tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            output_ids = self._local_llm.generate(**inputs, generation_config=gen_cfg)
        if not self._local_llm.config.is_encoder_decoder:
            output_ids = output_ids[:, len(inputs["input_ids"][0]) :]
        outputs = self._tokenizer.batch_decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )
        return outputs

    def rerank(
        self,
        query: str,
        chunks: list[str],
        num_passes: int = 3,
        window_size: int = 20,
        step: int = 10,
        top_k_candidates: int | None = None,
        device: str | torch.device | None = None,
    ) -> npt.NDArray[np.int32]:
        """
        Returns an array of indexes in `chunks` in descending order of relevance to the `query`.
        For example, if `rank` is output of this function,
        `chunks[rank[0]]` is the most relevant chunk, and `chunks[rank[1]]` is the second most relevant, and etc.
        """
        rank_start = 0
        rank_end = len(chunks) if top_k_candidates is None else min(top_k_candidates, len(chunks))
        window_size = min(window_size, rank_end - rank_start)
        rank = np.arange(rank_end - rank_start, dtype=np.int32)

        for _ in range(num_passes):
            end_pos = rank_end
            start_pos = rank_end - window_size
            # end_pos > rank_start ensures that the list is non-empty
            # while allowing last window to be smaller than window_size;
            # start_pos + step != rank_start prevents processing of redundant windows (e.g. 0-20, followed by 0-10)
            while end_pos > rank_start and start_pos + step != rank_start:
                start_pos = max(start_pos, rank_start)
                window_chunks = [chunks[i] for i in rank[start_pos:end_pos]]
                window_rank = self.permutation_pipeline(query, window_chunks, device=device)
                rank[start_pos:end_pos] = rank[start_pos:end_pos][window_rank]
                end_pos = end_pos - step
                start_pos = start_pos - step
        return rank

    def rerank_batch(
        self,
        queries: list[str],
        chunks: list[list[str]],  # each query must have the same number of chunks.
        batch_size: int = 1,
        num_passes: int = 3,
        window_size: int = 20,
        step: int = 10,
        top_k_candidates: int | None = None,
        device: str | torch.device | None = None,
    ) -> npt.NDArray[np.int32]:
        num_chunks = len(chunks[0])
        rank_start = 0
        rank_end = num_chunks if top_k_candidates is None else min(top_k_candidates, num_chunks)
        window_size = min(window_size, rank_end - rank_start)

        # (num_query, num_tank)
        rank = np.tile(np.arange(rank_end - rank_start, dtype=np.int32), (len(chunks), 1))
        num_batches = -(-len(queries) // batch_size)
        for batch_id in trange(num_batches, total=num_batches):
            batch_start = batch_id * batch_size
            batch_end = min(batch_start + batch_size, len(queries))
            for _ in range(num_passes):
                end_pos = rank_end
                start_pos = rank_end - window_size
                # end_pos > rank_start ensures that the list is non-empty
                # while allowing last window to be smaller than window_size;
                # start_pos + step != rank_start prevents processing of redundant windows (e.g. 0-20, followed by 0-10)
                while end_pos > rank_start and start_pos + step != rank_start:
                    start_pos = max(start_pos, rank_start)
                    window_chunks = [
                        [chunks[j][i] for i in rank[j, start_pos:end_pos]]
                        for j in range(batch_start, batch_end)
                    ]
                    window_rank = self.permutation_pipeline_batch(
                        queries[batch_start:batch_end], window_chunks, device=device
                    )
                    rank[batch_start:batch_end, start_pos:end_pos] = np.take_along_axis(
                        rank[batch_start:batch_end, start_pos:end_pos], window_rank, axis=1
                    )
                    end_pos = end_pos - step
                    start_pos = start_pos - step
        return rank

    def permutation_pipeline(
        self,
        query: str,
        chunks: list[str],
        device: str | torch.device | None = None,
    ) -> list[int]:
        prompt = self.create_prompt(query, chunks)
        permutation = self.run_llm(prompt, current_window_size=len(chunks), device=device)
        window_rank = self.parse_permutation(permutation, current_window_size=len(chunks))
        return window_rank

    def permutation_pipeline_batch(
        self,
        queries: list[str],
        chunks: list[list[str]],
        device: str | torch.device | None = None,
    ) -> npt.NDArray:
        prompts = [self.create_prompt(query, chunk) for query, chunk in zip(queries, chunks)]
        permutations = self.run_llm_batch(
            prompts, current_window_size=len(chunks[0]), device=device
        )
        window_ranks = [
            self.parse_permutation(permutation, current_window_size=len(chunks[0]))
            for permutation in permutations
        ]
        return np.asarray(window_ranks)

    def create_prompt(
        self,
        query: str,
        chunks: list[str],
    ) -> str:
        query = query
        num = len(chunks)

        # max length for each chunk. https://github.com/castorini/rank_llm/blob/fd4010914703b2ba9b3724d9aa7f0a6ce4195eec/src/rank_llm/rerank/rank_listwise_os_llm.py#L140  # noqa: E501
        # max_length = int(300 * (20 / len(chunks)))

        prompt = (
            "<|system|>\nYou are RankLLM, an intelligent assistant that can rank passages based on their relevancy"
            " to the query.</s>\n"
        )
        prefix = self._add_prefix_prompt(query, num)
        rank = 0
        input_context = f"{prefix}\n"
        for content in chunks:
            rank += 1
            content = content.replace("Title: Content: ", "")
            content = content.strip()
            content = " ".join(content.split())
            input_context += f"[{rank}] {self._replace_number(content)}\n"
        input_context += self._add_post_prompt(query, num)
        prompt += "<|user|>\n" + input_context + "</s>\n<|assistant|>\n"
        prompt = fix_text(prompt)
        num_tokens = self.get_num_tokens(prompt)
        if num_tokens > self._context_size - self.num_output_tokens(len(chunks)):
            raise ValueError(
                f"Prompt exceeds context size {num_tokens + self.num_output_tokens(len(chunks))} > {self._context_size}"
            )

        return prompt

    def parse_permutation(self, permutation: str, current_window_size: int) -> list[int]:
        response = self._clean_response(permutation)
        response = [int(x) - 1 for x in response.split()]
        response = self._remove_duplicate(response)
        original_rank = [tt for tt in range(current_window_size)]
        response = [ss for ss in response if ss in original_rank]
        response = response + [tt for tt in original_rank if tt not in response]
        return response

    def get_num_tokens(self, prompt: str) -> int:
        return len(self._tokenizer.encode(prompt))

    def num_output_tokens(self, current_window_size: int) -> int:
        num_output_token = self._num_output_token_cache.get(current_window_size)
        if num_output_token:
            return num_output_token
        num_output_token = (
            len(
                self._tokenizer.encode(" > ".join([f"[{i+1}]" for i in range(current_window_size)]))
            )
            - 1
        )
        self._num_output_token_cache[current_window_size] = num_output_token
        return num_output_token

    def _add_prefix_prompt(self, query: str, num: int) -> str:
        return (
            f"I will provide you with {num} passages, each indicated by a numerical identifier []."
            f" Rank the passages based on their relevance to the search query: {query}.\n"
        )

    def _add_post_prompt(self, query: str, num: int) -> str:
        example_ordering = "[2] > [1]"
        return (
            f"Search Query: {query}.\nRank the {num} passages above based on their relevance to the search query."
            f" All the passages should be included and listed using identifiers, in descending order of relevance."
            f" The output format should be [] > [], e.g., {example_ordering},"
            f" Only respond with the ranking results, do not say any word or explain."
        )

    def _replace_number(self, s: str) -> str:
        return re.sub(r"\[(\d+)\]", r"(\1)", s)

    def _clean_response(self, response: str) -> str:
        new_response = ""
        for c in response:
            if not c.isdigit():
                new_response += " "
            else:
                new_response += c
        new_response = new_response.strip()
        return new_response

    def _remove_duplicate(self, response: list[int]) -> list[int]:
        new_response = []
        for c in response:
            if c not in new_response:
                new_response.append(c)
        return new_response
