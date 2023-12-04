from transformers import (
    RobertaForMaskedLM,
    RobertaTokenizerFast,
)
import torch
import os

from civic.config import MODEL_STORAGE_DIR
from civic.utils.filesystem_utils import create_folder_if_not_exists


class RobertaLongForMaskedLM(RobertaForMaskedLM):
    @staticmethod
    def create_biomed_roberta_long_model(max_pos):
        model = RobertaForMaskedLM.from_pretrained("allenai/biomed_roberta_base")
        tokenizer = RobertaTokenizerFast.from_pretrained(
            "allenai/biomed_roberta_base",
            model_max_length=max_pos,
        )
        config = model.config

        # extend position embeddings
        tokenizer.model_max_length = max_pos
        tokenizer.init_kwargs["model_max_length"] = max_pos
        (
            current_max_pos,
            embed_size,
        ) = model.roberta.embeddings.position_embeddings.weight.shape
        max_pos += 2  # NOTE: RoBERTa has positions 0,1 reserved, so embedding size is max position + 2
        config.max_position_embeddings = max_pos
        assert max_pos > current_max_pos
        # allocate a larger position embedding matrix
        new_pos_embed = model.roberta.embeddings.position_embeddings.weight.new_empty(
            max_pos, embed_size
        )
        # copy position embeddings over and over to initialize the new position embeddings
        k = 2
        step = current_max_pos - 2
        while k < max_pos - 1:
            new_pos_embed[
                k : (k + step)
            ] = model.roberta.embeddings.position_embeddings.weight[2:]
            k += step
        model.roberta.embeddings.position_embeddings.weight.data = new_pos_embed
        model.roberta.embeddings.position_ids.data = torch.tensor(
            [i for i in range(max_pos)]
        ).reshape(1, max_pos)

        model.save_pretrained(os.path.join(MODEL_STORAGE_DIR, "biomed_roberta_base"))
        tokenizer.save_pretrained(
            os.path.join(MODEL_STORAGE_DIR, "biomed_roberta_base-1024")
        )
        return model, tokenizer

    @staticmethod
    def from_biomed_roberta_snapshot():
        if not create_folder_if_not_exists(
            os.path.join(MODEL_STORAGE_DIR, "biomed_roberta_base-1024")
        ):
            print("Creating Long Model...")
            RobertaLongForMaskedLM.create_biomed_roberta_long_model(1024)
            print("Finished Creating Long Model.")
        print("Loading model biomed_roberta_base-1024 from disk")
        tokenizer = RobertaTokenizerFast.from_pretrained(
            os.path.join(MODEL_STORAGE_DIR, "biomed_roberta_base-1024")
        )
        model = RobertaLongForMaskedLM.from_pretrained(
            os.path.join(MODEL_STORAGE_DIR, "biomed_roberta_base-1024")
        )
        return tokenizer, model

    def __init__(self, config):
        super().__init__(config)
