from transformers import LongformerTokenizer, LongformerForSequenceClassification


class Longformer:
    @staticmethod
    def from_longformer_allenai_base_pretrained():
        model_name = "allenai/longformer-base-4096"
        tokenizer = LongformerTokenizer.from_pretrained(model_name)
        model = LongformerForSequenceClassification.from_pretrained(
            model_name, num_labels=5
        )
        return tokenizer, model

    @staticmethod
    def from_roberta_snapshot():
        raise NotImplementedError
