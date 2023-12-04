from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
)


class BertForCivicEvidenceClassification:
    @staticmethod
    def _create_tokenizer_and_model_from_snapshot(
        snapshot_name, model_snapshot=None, num_labels=5
    ):
        tokenizer = BertTokenizerFast.from_pretrained(snapshot_name)
        model = BertForSequenceClassification.from_pretrained(
            snapshot_name if not model_snapshot else model_snapshot,
            num_labels=num_labels,
        )
        return tokenizer, model

    @staticmethod
    def from_bert_base_uncased(model_snapshot=None):
        return BertForCivicEvidenceClassification._create_tokenizer_and_model_from_snapshot(
            "bert-base-uncased", model_snapshot=model_snapshot
        )

    @staticmethod
    def from_pubmed_bert(model_snapshot=None):
        return BertForCivicEvidenceClassification._create_tokenizer_and_model_from_snapshot(
            "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
            model_snapshot=model_snapshot,
        )

    @staticmethod
    def from_bio_link_bert(model_snapshot=None):
        return BertForCivicEvidenceClassification._create_tokenizer_and_model_from_snapshot(
            "michiyasunaga/BioLinkBERT-base", model_snapshot=model_snapshot
        )

    @staticmethod
    def from_bio_link_bert_large(model_snapshot=None):
        return BertForCivicEvidenceClassification._create_tokenizer_and_model_from_snapshot(
            "michiyasunaga/BioLinkBERT-large", model_snapshot=model_snapshot
        )
