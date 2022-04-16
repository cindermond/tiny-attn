from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from rewriter.model.roberta_bl import RobertaModelBL
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from rewriter.model.RobertaSCHead import RobertaSCHead
import torch

class RobertaForSC(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, output_nlayers, is_rewriter, rewriter_nhead, rewriter_d_hid, rewriter_dropout, rewriter_nlayers, n_labels, attention_emd = 64, attention_head = 1, structure = 'seq'):
        super().__init__(config)
        self.num_labels = n_labels
        self.config = config

        self.roberta = RobertaModelBL(config, is_rewriter=is_rewriter,rewriter_nhead=rewriter_nhead, rewriter_d_hid=rewriter_d_hid, rewriter_dropout=rewriter_dropout, rewriter_nlayers=rewriter_nlayers, attention_emd=attention_emd, attention_head=attention_head, structure=structure)
        self.classifier = RobertaSCHead(config, output_nlayers=output_nlayers, num_labels=n_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )