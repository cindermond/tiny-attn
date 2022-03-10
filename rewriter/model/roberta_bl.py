from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaEmbeddings, RobertaLayer
import torch
from transformers.modeling_outputs import BaseModelOutput
from torch import nn

class TinyAttention(nn.Module):
    def __init__(self, input_embd=1024, output_embd=1024, attention_embd=64, attention_head=1, attention_dropout=0.1) -> None:
        super().__init__()
        self.attention_embd = attention_embd
        self.linear1 = nn.Linear(input_embd, attention_embd * 3)
        self.attention = nn.MultiheadAttention(attention_embd, attention_head, attention_dropout, batch_first= True)
        self.linear2 = nn.Linear(attention_embd, output_embd)
        self.norm = nn.LayerNorm(input_embd)

    def forward(self, hidden_states):
        new_hs = self.norm(hidden_states)
        new_hs = self.linear1(new_hs)
        q,k,v = torch.split(new_hs,self.attention_embd, dim=2)
        new_hs = self.attention(q,k,v)[0]
        new_hs = self.linear2(new_hs)
        return hidden_states + new_hs*0.01

    

class RobertaEncoderBL(nn.Module):
    def __init__(self, config, input_embd=1024, output_embd=1024, attention_embd=64, attention_head=1, attention_dropout=0.1):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([RobertaLayer(config) for _ in range(config.num_hidden_layers)])
        self.attention_layer = nn.ModuleList([TinyAttention(input_embd=input_embd, output_embd=output_embd, attention_embd=attention_embd, attention_head=attention_head, attention_dropout=attention_dropout) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
    ):
        for attention_module, layer_module in zip(self.attention_layer, self.layer):
            
            hidden_states = attention_module(
                hidden_states
            )

            layer_outputs = layer_module(
                hidden_states,
                attention_mask
            )

            hidden_states = layer_outputs[0]
        return BaseModelOutput(
            last_hidden_state=hidden_states,
        )


class RobertaModelBL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, is_rewriter=True, rewriter_nhead=4, rewriter_d_hid=512, rewriter_dropout=0.1, rewriter_nlayers=1):
        super().__init__(config)
        self.config = config

        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoderBL(config)
        self.first_dropout = torch.nn.Dropout(0.1)
        self.is_rewriter = is_rewriter
        if is_rewriter:
            self.rewriter = nn.TransformerEncoder(nn.TransformerEncoderLayer(1024, rewriter_nhead, rewriter_d_hid, rewriter_dropout, batch_first=True), rewriter_nlayers)

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
        )
        if self.is_rewriter:
            embedding_output = self.first_dropout(embedding_output)
            embedding_output = self.rewriter(embedding_output)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
        )
        sequence_output = encoder_outputs[0]

        return BaseModelOutput(
            last_hidden_state=sequence_output
        )