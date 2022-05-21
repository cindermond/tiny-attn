from transformers.models.albert.modeling_albert import AlbertPreTrainedModel, AlbertEmbeddings, AlbertAttention
from transformers.models.albert.configuration_albert import AlbertConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling, BaseModelOutput
from transformers.activations import ACT2FN
from transformers.pytorch_utils import apply_chunking_to_forward
import torch
from torch import nn
from typing import List, Dict, Optional, Union, Tuple
from rewriter.model.tiny_attn import TinyAttention

class AlbertLayerBL(nn.Module):
    def __init__(self, config: AlbertConfig):
        super().__init__()

        self.config = config
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.full_layer_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = AlbertAttention(config)
        self.ffn = nn.Linear(config.hidden_size, config.intermediate_size)
        self.ffn_output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = ACT2FN[config.hidden_act]
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_output = self.attention(hidden_states, attention_mask, head_mask, output_attentions)

        ffn_output = apply_chunking_to_forward(
            self.ff_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output[0],
        )
        hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])

        return (hidden_states,) + attention_output[1:]  # add attentions if we output them

    def ff_chunk(self, attention_output: torch.Tensor) -> torch.Tensor:
        ffn_output = self.ffn(attention_output)
        ffn_output = self.activation(ffn_output)
        ffn_output = self.ffn_output(ffn_output)
        return ffn_output

class AlbertLayerGroupBL(nn.Module):
    def __init__(self, config: AlbertConfig):
        super().__init__()

        self.albert_layers = nn.ModuleList([AlbertLayerBL(config) for _ in range(config.inner_group_num)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        layer_hidden_states = ()
        layer_attentions = ()

        for layer_index, albert_layer in enumerate(self.albert_layers):
            layer_output = albert_layer(hidden_states, attention_mask, head_mask[layer_index], output_attentions)
            hidden_states = layer_output[0]

            if output_attentions:
                layer_attentions = layer_attentions + (layer_output[1],)

            if output_hidden_states:
                layer_hidden_states = layer_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (layer_hidden_states,)
        if output_attentions:
            outputs = outputs + (layer_attentions,)
        return outputs  # last-layer hidden state, (layer hidden states), (layer attentions)

class AlbertTransformerBL(nn.Module):
    def __init__(self, config: AlbertConfig):
        super().__init__()

        self.config = config
        self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)
        self.albert_layer_groups = nn.ModuleList([AlbertLayerGroupBL(config) for _ in range(config.num_hidden_groups)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[BaseModelOutput, Tuple]:
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)

        all_hidden_states = (hidden_states,) if output_hidden_states else None
        all_attentions = () if output_attentions else None

        head_mask = [None] * self.config.num_hidden_layers if head_mask is None else head_mask

        for i in range(self.config.num_hidden_layers):
            # Number of layers in a hidden group
            layers_per_group = int(self.config.num_hidden_layers / self.config.num_hidden_groups)

            # Index of the hidden group
            group_idx = int(i / (self.config.num_hidden_layers / self.config.num_hidden_groups))

            layer_group_output = self.albert_layer_groups[group_idx](
                hidden_states,
                attention_mask,
                head_mask[group_idx * layers_per_group : (group_idx + 1) * layers_per_group],
                output_attentions,
                output_hidden_states,
            )
            hidden_states = layer_group_output[0]

            if output_attentions:
                all_attentions = all_attentions + layer_group_output[-1]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


class AlbertModelBL(AlbertPreTrainedModel):

    config_class = AlbertConfig
    base_model_prefix = "albert"

    def __init__(self, config: AlbertConfig, add_pooling_layer: bool = True):
        super().__init__(config)

        self.config = config
        self.embeddings = AlbertEmbeddings(config)
        self.encoder = AlbertTransformerBL(config)
        if add_pooling_layer:
            self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
            self.pooler_activation = nn.Tanh()
        else:
            self.pooler = None
            self.pooler_activation = None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} ALBERT has
        a different architecture in that its layers are shared across groups, which then has inner groups. If an ALBERT
        model has 12 hidden layers and 2 hidden groups, with two inner groups, there is a total of 4 different layers.
        These layers are flattened: the indices [0,1] correspond to the two inner groups of the first hidden layer,
        while [2,3] correspond to the two inner groups of the second hidden layer.
        Any layer with in index other than [0,1,2,3] will result in an error. See base class PreTrainedModel for more
        information about head pruning
        """
        for layer, heads in heads_to_prune.items():
            group_idx = int(layer / self.config.inner_group_num)
            inner_group_idx = int(layer - group_idx * self.config.inner_group_num)
            self.encoder.albert_layer_groups[group_idx].albert_layers[inner_group_idx].attention.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[None] = None,
        output_hidden_states: Optional[None] = None,
        return_dict: Optional[None] = None,
    ) -> Union[BaseModelOutputWithPooling, Tuple]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]

        pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0])) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )