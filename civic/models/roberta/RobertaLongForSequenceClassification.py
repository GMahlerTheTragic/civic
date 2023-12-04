from transformers import RobertaForSequenceClassification


class RobertaLongForSequenceClassification(RobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
