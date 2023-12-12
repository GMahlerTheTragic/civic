import torch
from captum.attr import LayerIntegratedGradients

from enum import Enum


class IntegratedGradientsModelType(Enum):
    Bert = 1
    Roberta = 2
    RobertaLong = 3


class IntegratedGradientWrapper:
    def __init__(self, model, integrated_gradients_model_type):
        self.model = model
        if integrated_gradients_model_type == IntegratedGradientsModelType.Bert:
            self.lig = LayerIntegratedGradients(
                self.forward_for_ig, self.model.bert.embeddings
            )
        elif integrated_gradients_model_type == IntegratedGradientsModelType.Roberta:
            self.lig = LayerIntegratedGradients(
                self.forward_for_ig, self.model.roberta.embeddings
            )
        else:
            raise ValueError(f"{type} is not a valid option.")

    def forward_for_ig(self, input_ids, attention_mask):
        logits = self.model(input_ids, attention_mask=attention_mask).logits
        return logits

    def do_attribution(self, input_ids, ref_ids, attention_mask, target):
        attributions = self.lig.attribute(
            input_ids.long(),
            ref_ids.long(),
            target=target,
            additional_forward_args=(attention_mask,),
        )
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        return attributions
