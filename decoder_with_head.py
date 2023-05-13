from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.utils.import_utils import is_torch_fx_proxy


class DecoderWithHead(nn.Module):
    def __init__(self, decoder, lm_head):
        super(DecoderWithHead, self).__init__()
        self.decoder = decoder
        self.lm_head = lm_head

        self.model_parallel = False

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.decoder.config.decoder_start_token_id
        pad_token_id = self.decoder.config.pad_token_id

        assert decoder_start_token_id is not None, (
            "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id."
            " See T5 docs for more information"
        )

        # shift inputs to the right
        if is_torch_fx_proxy(input_ids):
            # Item assignment is not supported natively for proxies.
            shifted_input_ids = torch.full(
                input_ids.shape[:-1] + (1,), decoder_start_token_id
            )
            shifted_input_ids = torch.cat(
                [shifted_input_ids, input_ids[..., :-1]], dim=-1
            )
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        assert (
            pad_token_id is not None
        ), "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

    def forward(
        self,
        hidden_states: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        if (
            labels is not None
            and decoder_input_ids is None
            and decoder_inputs_embeds is None
        ):
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]
        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        # if not return_dict:
        #     output = (lm_logits,) + decoder_outputs[1:]
        #     return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
        )
