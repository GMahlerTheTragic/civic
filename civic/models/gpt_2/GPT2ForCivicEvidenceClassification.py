from transformers import GPT2Tokenizer, GPT2ForSequenceClassification


class GPT2ForCivicEvidenceClassification:
    @staticmethod
    def from_biomed_lm_snapshot():
        model_name = "stanford-crfm/BioMedLM"
        tokenizer = GPT2Tokenizer.from_pretrained(model_name, pad_token="<|endoftext|>")
        model = GPT2ForSequenceClassification.from_pretrained(model_name, num_labels=5)
        tokenizer.pad_token_id = 50256
        model.config.pad_token_id = 50256
        return tokenizer, model
