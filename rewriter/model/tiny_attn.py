import torch
from torch import nn
from transformers.models.bart.modeling_bart import BartAttention


class TinyAttention_(nn.Module):
    #an old version, past weights need this.
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
        return hidden_states + 0.01*new_hs

class TinyAttention(nn.Module):
    def __init__(self, input_embd=1024, output_embd=1024, attention_embd=64, attention_head=1, attention_dropout=0.1) -> None:
        super().__init__()
        self.attention_embd = attention_embd
        self.linear1 = nn.Linear(input_embd, attention_embd * 3)
        self.attention = nn.MultiheadAttention(attention_embd, attention_head, attention_dropout, batch_first= True)
        self.linear2 = nn.Linear(attention_embd, output_embd)
        self.norm = nn.LayerNorm(input_embd)
        with torch.no_grad():
            for p in self.linear2.parameters():
                p *= 0.01

    def forward(self, hidden_states, attention_mask = None):
        new_hs = self.norm(hidden_states)
        new_hs = self.linear1(new_hs)
        q,k,v = torch.split(new_hs,self.attention_embd, dim=2)
        if attention_mask is None:
            new_hs = self.attention(q,k,v)[0]
        else:
            new_hs = self.attention(q,k,v, key_padding_mask=torch.logical_not(attention_mask))[0]
        new_hs = self.linear2(new_hs)
        return new_hs


class BartTinyAttention(nn.Module):
    def __init__(self, input_embd=1024, output_embd=1024, attention_embd=64, attention_head=1, attention_dropout=0.1) -> None:
        super().__init__()
        self.attention_embd = attention_embd
        self.linear1 = nn.Linear(input_embd, attention_embd)
        self.attention = BartAttention(attention_embd, attention_head, attention_dropout, is_decoder=True)
        self.linear2 = nn.Linear(attention_embd, output_embd)
        self.norm = nn.LayerNorm(input_embd)
        with torch.no_grad():
            for p in self.linear2.parameters():
                p *= 0.01
            

    def forward(self, hidden_states, encoder_hidden_states=None, past_key_values = None, attention_mask = None, encoder_attention_mask = None, **kwargs):
        new_hs = self.norm(hidden_states)
        new_hs = self.linear1(new_hs)
        new_hs, _, past_key_value = self.attention(new_hs, key_value_states=encoder_hidden_states, past_key_value = past_key_values, attention_mask=encoder_attention_mask)
        new_hs = self.linear2(new_hs)
        return (new_hs, past_key_value)