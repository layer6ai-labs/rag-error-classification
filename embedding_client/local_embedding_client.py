import logging

import torch
from tqdm import tqdm
from transformers import AutoModel
from transformers import AutoTokenizer

from baseconfig import EmbeddingConfig


class LocalEmbeddingClient:
    def __init__(self, config: EmbeddingConfig):
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.info("Initiating Embedding Client...")
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, device_map=config.device)
        self.model = AutoModel.from_pretrained(
            config.model, trust_remote_code=True, device_map=config.device
        )
        self.model.eval()
        self.log.info("Done")

    def encode(self, text, batch_size, device=None) -> torch.Tensor:
        device = device if device is not None else self.config.device
        self.model.to(device)

        single_text = False
        if isinstance(text, str):
            text = [text]
            single_text = True

        all_embeddings = []
        num_batches = -(-len(text) // batch_size)
        with torch.no_grad():
            for i in tqdm(range(num_batches), total=num_batches):
                start = i * batch_size
                end = min(start + batch_size, len(text))
                truncated_text = text[start:end]
                dic = self.tokenizer(
                    truncated_text, return_tensors="pt", truncation=True, padding=True
                ).to(device)
                outputs = self.model(**dic)
                embeddings = outputs[0][:, 0]
                all_embeddings.append(embeddings.detach().cpu())
        embeddings = torch.concatenate(all_embeddings, dim=0)

        if single_text:
            return embeddings[0]
        return embeddings
