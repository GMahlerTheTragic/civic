from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    LongformerTokenizerFast,
    LongformerForSequenceClassification,
)

from civic.models.roberta.RobertaLongForSequenceClassification import (
    RobertaLongForSequenceClassification,
)


class RobertaForCivicEvidenceClassification:
    @staticmethod
    def _create_roberta_tokenizer_and_model_from_snapshot(
        snapshot_name, model_snapshot=None, num_labels=5
    ):
        tokenizer = RobertaTokenizerFast.from_pretrained(snapshot_name)
        model = RobertaForSequenceClassification.from_pretrained(
            snapshot_name if not model_snapshot else model_snapshot,
            num_labels=num_labels,
        )
        return tokenizer, model

    @staticmethod
    def from_roberta_base(model_snapshot=None):
        return RobertaForCivicEvidenceClassification._create_roberta_tokenizer_and_model_from_snapshot(
            "roberta-base", model_snapshot=model_snapshot
        )

    @staticmethod
    def from_biomed_roberta_base(model_snapshot=None):
        return RobertaForCivicEvidenceClassification._create_roberta_tokenizer_and_model_from_snapshot(
            "allenai/biomed_roberta_base", model_snapshot=model_snapshot
        )

    @staticmethod
    def from_longformer_base(model_snapshot=None):
        model_name = "allenai/longformer-base-4096"
        tokenizer = LongformerTokenizerFast.from_pretrained(model_name)
        model = LongformerForSequenceClassification.from_pretrained(
            model_name if not model_snapshot else model_snapshot, num_labels=5
        )
        return tokenizer, model

    @staticmethod
    def from_long_biomed_roberta_pretrained(model_snapshot=None):
        model_name = "tmp2/checkpoint-3000"
        tokenizer = RobertaTokenizerFast.from_pretrained("allenai/biomed_roberta_base")
        model = RobertaLongForSequenceClassification.from_pretrained(
            model_name if not model_snapshot else model_snapshot, num_labels=5
        )
        return tokenizer, model
