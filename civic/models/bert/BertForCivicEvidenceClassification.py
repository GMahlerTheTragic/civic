from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
)


class BertForCivicEvidenceClassification:
    @staticmethod
    def _create_tokenizer_and_model_from_snapshot(snapshot_name, num_labels=5):
        tokenizer = BertTokenizerFast.from_pretrained(snapshot_name)
        model = BertForSequenceClassification.from_pretrained(
            snapshot_name, num_labels=num_labels
        )
        return tokenizer, model

    @staticmethod
    def from_bert_base_uncased():
        return BertForCivicEvidenceClassification._create_tokenizer_and_model_from_snapshot(
            "bert-base-uncased"
        )

    @staticmethod
    def from_pubmed_bert():
        return BertForCivicEvidenceClassification._create_tokenizer_and_model_from_snapshot(
            "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
        )

    @staticmethod
    def from_bio_link_bert():
        return BertForCivicEvidenceClassification._create_tokenizer_and_model_from_snapshot(
            "michiyasunaga/BioLinkBERT-base"
        )
