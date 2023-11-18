from transformers import LongformerSelfAttention


class BertLongSelfAttention(LongformerSelfAttention):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):
        return super().forward(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
