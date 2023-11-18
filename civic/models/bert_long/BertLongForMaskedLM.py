import copy

from transformers import BertForMaskedLM, BertTokenizerFast, LongformerSelfAttention
import torch

from civic.models.bert_long.BertLongSelfAttention import BertLongSelfAttention
from civic.utils.filesystem_utils import check_file_exists


class BertLongForMaskedLM(BertForMaskedLM):
    @staticmethod
    def create_biomed_bert_long_model(attention_window, max_pos):
        model = BertForMaskedLM.from_pretrained(
            "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
        )
        tokenizer = BertTokenizerFast.from_pretrained(
            "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
            model_max_length=max_pos,
        )
        config = model.config

        # extend position embeddings
        tokenizer.model_max_length = max_pos
        tokenizer.init_kwargs["model_max_length"] = max_pos
        (
            current_max_pos,
            embed_size,
        ) = model.bert.embeddings.position_embeddings.weight.shape

        config.max_position_embeddings = max_pos
        assert max_pos > current_max_pos
        # allocate a larger position embedding matrix
        new_pos_embed = model.bert.embeddings.position_embeddings.weight.new_empty(
            max_pos, embed_size
        )
        # copy position embeddings over and over to initialize the new position embeddings
        k = 0
        step = current_max_pos
        while k < max_pos - 1:
            new_pos_embed[
                k : (k + step)
            ] = model.bert.embeddings.position_embeddings.weight[0:]
            k += step
        model.bert.embeddings.position_embeddings.weight.data = new_pos_embed
        model.bert.embeddings.position_ids.data = torch.tensor(
            [i for i in range(max_pos)]
        ).reshape(1, max_pos)

        # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
        config.attention_window = [attention_window] * config.num_hidden_layers
        for i, layer in enumerate(model.bert.encoder.layer):
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

        model.save_pretrained("BiomedNLP-BiomedBERT-base-uncased-abstract-4096")
        tokenizer.save_pretrained("BiomedNLP-BiomedBERT-base-uncased-abstract-4096")
        return model, tokenizer

    @staticmethod
    def from_biobert_snapshot():
        if not check_file_exists("BiomedNLP-BiomedBERT-base-uncased-abstract-4096"):
            BertLongForMaskedLM.create_biomed_bert_long_model(512, 4096)
        print("Loading the model from BiomedNLP-BiomedBERT-base-uncased-abstract-4096")
        tokenizer = BertTokenizerFast.from_pretrained(
            "BiomedNLP-BiomedBERT-base-uncased-abstract-4096"
        )
        model = BertLongForMaskedLM.from_pretrained(
            "BiomedNLP-BiomedBERT-base-uncased-abstract-4096"
        )
        return (
            tokenizer,
            model,
        )

    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.bert.encoder.layer):
            # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
            layer.attention.self = BertLongSelfAttention(config, layer_id=i)
