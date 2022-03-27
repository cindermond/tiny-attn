import torch
from torch import nn
from transformers.models.bart.modeling_bart import BartAttention

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

class BartTinyAttention(nn.Module):
    def __init__(self, input_embd=1024, output_embd=1024, attention_embd=64, attention_head=1, attention_dropout=0.1, code=0) -> None:
        super().__init__()
        self.attention_embd = attention_embd
        self.linear1 = nn.Linear(input_embd, attention_embd)
        self.attention = BartAttention(attention_embd, attention_head, attention_dropout, is_decoder=True)
        self.linear2 = nn.Linear(attention_embd, output_embd)
        self.norm = nn.LayerNorm(input_embd)
        self.code = code

    def forward(self, hidden_states, encoder_hidden_states=None, past_key_values = None, attention_mask = None, encoder_attention_mask = None, **kwargs):
        new_hs = self.norm(hidden_states)
        new_hs = self.linear1(new_hs)
        if self.code[0] == '2':
            new_hs, _, past_key_value = self.attention(new_hs, past_key_value = past_key_values, attention_mask=attention_mask)
        else:
            new_hs, _, past_key_value = self.attention(new_hs, key_value_states=encoder_hidden_states, past_key_value = past_key_values, attention_mask=encoder_attention_mask)
        new_hs = self.linear2(new_hs)
        if self.code[2] == '1':
            (hidden_states + new_hs*0.001, past_key_value)
        return (hidden_states + new_hs*0.01, past_key_value)

class BartDoubleTinyAttention(nn.Module):
    def __init__(self, input_embd=1024, output_embd=1024, attention_embd=64, attention_head=1, attention_dropout=0.1, code = 0) -> None:
        super().__init__()
        self.attention_embd = attention_embd
        self.linear1 = nn.Linear(input_embd, attention_embd)
        self.attention1 = BartAttention(attention_embd, attention_head, attention_dropout, is_decoder=True)
        self.attention2 = BartAttention(attention_embd, attention_head, attention_dropout, is_decoder=True)
        self.linear2 = nn.Linear(attention_embd, output_embd)
        self.norm = nn.LayerNorm(input_embd)
        self.code = code

    def forward(self, hidden_states, encoder_hidden_states=None, past_key_values = None, attention_mask = None, encoder_attention_mask = None, **kwargs):
        new_hs = self.norm(hidden_states)
        new_hs = self.linear1(new_hs)
        if past_key_values is None:
            new_hs, _, past_key_value1 = self.attention1(new_hs, key_value_states=encoder_hidden_states, past_key_value = None, attention_mask=encoder_attention_mask)
            new_hs, _, past_key_value2 = self.attention2(new_hs, past_key_value = None, attention_mask=attention_mask)
        else:
            new_hs, _, past_key_value1 = self.attention1(new_hs, key_value_states=encoder_hidden_states, past_key_value = past_key_values[:1], attention_mask=encoder_attention_mask)
            new_hs, _, past_key_value2 = self.attention2(new_hs, past_key_value = past_key_values[1:], attention_mask=attention_mask)
        new_hs = self.linear2(new_hs)
        past_key_value = past_key_value1 + past_key_value2
        if self.code[2] == '1':
            return (hidden_states + new_hs*0.001, past_key_value)
        return (hidden_states + new_hs*0.01, past_key_value)