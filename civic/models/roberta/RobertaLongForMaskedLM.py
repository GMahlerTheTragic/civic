import copy
from transformers import (
    LongformerSelfAttention,
    RobertaForMaskedLM,
    RobertaTokenizerFast,
)
import torch
import os

from civic.config import MODEL_STORAGE_DIR
from civic.models.roberta.RobertaLongSelfAttention import RobertaLongSelfAttention
from civic.utils.filesystem_utils import create_folder_if_not_exists


class RobertaLongForMaskedLM(RobertaForMaskedLM):
    @staticmethod
    def create_biomed_roberta_long_model(attention_window, max_pos):
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

        # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
        config.attention_window = [attention_window] * config.num_hidden_layers
        for i, layer in enumerate(model.roberta.encoder.layer):
            longformer_self_attn = LongformerSelfAttention(config, layer_id=i)
            longformer_self_attn.query = layer.attention.self.query
            longformer_self_attn.key = layer.attention.self.key
            longformer_self_attn.value = layer.attention.self.value

            longformer_self_attn.query_global = copy.deepcopy(
                layer.attention.self.query
            )
            longformer_self_attn.key_global = copy.deepcopy(layer.attention.self.key)
            longformer_self_attn.value_global = copy.deepcopy(
                layer.attention.self.value
            )

            layer.attention.self = longformer_self_attn

        model.save_pretrained(
            os.path.join(MODEL_STORAGE_DIR, "biomed_roberta_base-1024")
        )
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
            RobertaLongForMaskedLM.create_biomed_roberta_long_model(512, 1024)
            print("Finished Creating Long Model.")
        print("Loading model biomed_roberta_base-1024 from disk")
        tokenizer = RobertaTokenizerFast.from_pretrained(
            os.path.join(MODEL_STORAGE_DIR, "biomed_roberta_base-1024")
        )
        model = RobertaForMaskedLM.from_pretrained(
            os.path.join(MODEL_STORAGE_DIR, "biomed_roberta_base-1024")
        )
        return tokenizer, model

    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.roberta.encoder.layer):
            # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
            layer.attention.self = RobertaLongSelfAttention(config, layer_id=i)
