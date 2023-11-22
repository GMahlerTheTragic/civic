from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    LongformerTokenizerFast,
    LongformerForSequenceClassification,
)


class RobertaForCivicEvidenceClassification:
    @staticmethod
    def _create_roberta_tokenizer_and_model_from_snapshot(snapshot_name, num_labels=5):
        tokenizer = RobertaTokenizerFast.from_pretrained(snapshot_name)
        model = RobertaForSequenceClassification.from_pretrained(
            snapshot_name, num_labels=num_labels
        )
        return tokenizer, model

    @staticmethod
    def from_roberta_base():
        return RobertaForCivicEvidenceClassification._create_roberta_tokenizer_and_model_from_snapshot(
            "roberta-base"
        )

    @staticmethod
    def from_biomed_roberta_base():
        return RobertaForCivicEvidenceClassification._create_roberta_tokenizer_and_model_from_snapshot(
            "allenai/biomed_roberta_base"
        )

    @staticmethod
    def from_longformer_base():
        model_name = "allenai/longformer-base-4096"
        tokenizer = LongformerTokenizerFast.from_pretrained(model_name)
        model = LongformerForSequenceClassification.from_pretrained(
            model_name, num_labels=5
        )
        return tokenizer, model
