from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaEmbeddings, RobertaLayer, RobertaAttention, RobertaIntermediate, RobertaOutput
import torch
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import apply_chunking_to_forward
from torch import nn
from rewriter.model.tiny_attn import TinyAttention_, TinyAttention

class RobertaLayerBL(nn.Module):
    def __init__(self, config, input_embd=768, output_embd=768, attention_embd=64, attention_head=1, attention_dropout=0.1, structure = 's0'):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = RobertaAttention(config)
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)
        self.tiny_attn = TinyAttention(input_embd=input_embd, output_embd=output_embd, attention_embd=attention_embd, attention_head=attention_head, attention_dropout=attention_dropout)
        self.structure = structure

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        old_attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        if self.structure[0]=='s':
            tiny_attn = self.tiny_attn(hidden_states, old_attention_mask)
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.structure[0]=='m':
            tiny_attn = self.tiny_attn(attention_output, old_attention_mask)
        if self.structure[1]=='0':
            attention_output = attention_output + tiny_attn

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )

        if self.structure[1]=='1':
            layer_output = layer_output + tiny_attn

        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output



class RobertaEncoderBL(nn.Module):
    def __init__(self, config, input_embd=768, output_embd=768, attention_embd=64, attention_head=1, attention_dropout=0.1, is_old_fashion=False, structure = 'seq'):
        super().__init__()
        self.config = config        
        self.is_old_fashion = is_old_fashion
        self.structure = structure
        if structure == 'seq':
            self.layer = nn.ModuleList([RobertaLayer(config) for _ in range(config.num_hidden_layers)])
            if is_old_fashion:
                self.attention_layer = nn.ModuleList([TinyAttention_(input_embd=input_embd, output_embd=output_embd, attention_embd=attention_embd, attention_head=attention_head, attention_dropout=attention_dropout) for _ in range(config.num_hidden_layers)])
            else:
                self.attention_layer = nn.ModuleList([TinyAttention(input_embd=input_embd, output_embd=output_embd, attention_embd=attention_embd, attention_head=attention_head, attention_dropout=attention_dropout) for _ in range(config.num_hidden_layers)])
        else:
            self.layer = nn.ModuleList([RobertaLayerBL(config, input_embd=input_embd, output_embd=output_embd, attention_embd=attention_embd, attention_head=attention_head, attention_dropout=attention_dropout, structure=structure) for _ in range(config.num_hidden_layers)])
    def forward(
        self,
        hidden_states,
        old_attention_mask=None,
        attention_mask=None,
    ):
        if self.structure == 'seq':
            for attention_module, layer_module in zip(self.attention_layer, self.layer):
                
                if self.is_old_fashion:
                    hidden_states = attention_module(
                        hidden_states
                    )
                else:
                    hidden_states = hidden_states + attention_module(
                        hidden_states,
                        old_attention_mask
                    )

                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask
                )

                hidden_states = layer_outputs[0]
        else:
            for layer_module in self.layer:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    old_attention_mask
                )

                hidden_states = layer_outputs[0]
        return BaseModelOutput(
            last_hidden_state=hidden_states,
        )


class RobertaModelBL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, is_rewriter=True, rewriter_nhead=4, rewriter_d_hid=512, rewriter_dropout=0.1, rewriter_nlayers=1, attention_emd = 64, attention_head = 1, structure = 'seq'):
        super().__init__(config)
        self.config = config

        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoderBL(config, input_embd=attention_emd, output_embd=attention_emd, attention_head=attention_head, structure=structure)
        self.first_dropout = torch.nn.Dropout(0.1)
        self.is_rewriter = is_rewriter
        if is_rewriter:
            self.rewriter = nn.TransformerEncoder(nn.TransformerEncoderLayer(config.hidden_size, rewriter_nhead, rewriter_d_hid, rewriter_dropout, batch_first=True), rewriter_nlayers)

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
            old_attention_mask = attention_mask,
            attention_mask=extended_attention_mask,
        )
        sequence_output = encoder_outputs[0]

        return BaseModelOutput(
            last_hidden_state=sequence_output
        )