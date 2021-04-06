from transformers.models.bert.modeling_bert import *
from torch.autograd import Variable
from torchblocks.losses import FocalLoss
from torch.nn import functional as F
from .modeling_util import concat_all_encoders_hidden_states


def masked_softmax(vector, seq_lens):
    mask = vector.new(vector.size()).zero_()
    for i in range(seq_lens.size(0)):
        mask[i, :, :seq_lens[i]] = 1
    mask = Variable(mask, requires_grad=False)
    # mask = None
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=-1)
    else:
        result = torch.nn.functional.softmax(vector * mask, dim=-1)
        result = result * mask
        result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)
    return result


class FuseNet(nn.Module):
    def __init__(self, config):
        super(FuseNet, self).__init__()
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear2 = nn.Linear(2 * config.hidden_size, 2 * config.hidden_size)

    def forward(self, inputs):
        p, q = inputs
        lq = self.linear(q)
        lp = self.linear(p)
        mid = nn.Sigmoid()(lq + lp)
        output = p * mid + q * (1 - mid)
        return output


class SSingleMatchNet(nn.Module):
    def __init__(self, config):
        super(SSingleMatchNet, self).__init__()
        self.map_linear = nn.Linear(2 * config.hidden_size, 2 * config.hidden_size)
        self.trans_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.drop_module = nn.Dropout(2 * config.hidden_dropout_prob)
        self.rank_module = nn.Linear(config.hidden_size * 2, 1)

    def forward(self, inputs):
        proj_p, proj_q, seq_len = inputs
        trans_q = self.trans_linear(proj_q)
        att_weights = proj_p.bmm(torch.transpose(trans_q, 1, 2))
        att_norm = masked_softmax(att_weights, seq_len)

        att_vec = att_norm.bmm(proj_q)
        output = nn.ReLU()(self.trans_linear(att_vec))
        return output


def seperate_seq(sequence_output, doc_len, ques_len, option_len):
    doc_seq_output = sequence_output.new(sequence_output.size()).zero_()
    doc_ques_seq_output = sequence_output.new(sequence_output.size()).zero_()
    ques_seq_output = sequence_output.new(sequence_output.size()).zero_()
    ques_option_seq_output = sequence_output.new(sequence_output.size()).zero_()
    option_seq_output = sequence_output.new(sequence_output.size()).zero_()
    for i in range(doc_len.size(0)):
        doc_seq_output[i, :doc_len[i]] = sequence_output[i, 1:doc_len[i] + 1]
        doc_ques_seq_output[i, :doc_len[i] + ques_len[i]] = sequence_output[i, :doc_len[i] + ques_len[i]]
        ques_seq_output[i, :ques_len[i]] = sequence_output[i, doc_len[i] + 2:doc_len[i] + ques_len[i] + 2]
        ques_option_seq_output[i, :ques_len[i] + option_len[i]] = sequence_output[i,
                                                                  doc_len[i] + 1:doc_len[i] + ques_len[i] + option_len[
                                                                      i] + 1]
        option_seq_output[i, :option_len[i]] = sequence_output[i,
                                               doc_len[i] + ques_len[i] + 2:doc_len[i] + ques_len[i] + option_len[
                                                   i] + 2]
    return doc_ques_seq_output, ques_option_seq_output, doc_seq_output, ques_seq_output, option_seq_output


class BertForMultipleChoiceWithMatch(BertPreTrainedModel):

    def __init__(self, config, num_choices=2):
        super(BertForMultipleChoiceWithMatch, self).__init__(config)
        self.num_choices = num_choices
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.classifier2 = nn.Linear(2 * config.hidden_size, 1)
        self.classifier3 = nn.Linear(3 * config.hidden_size, 1)
        self.classifier4 = nn.Linear(4 * config.hidden_size, 1)
        self.classifier6 = nn.Linear(6 * config.hidden_size, 1)
        self.ssmatch = SSingleMatchNet(config)
        self.pooler = BertPooler(config)
        self.fuse = FuseNet(config)
        # self.apply(self.init_bert_weights)
        self.init_weights()

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, doc_len=None, ques_len=None,
                option_len=None, labels=None, is_3=False, return_dict=None):

        r"""
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        doc_len = doc_len.view(-1, doc_len.size(0) * doc_len.size(1)).squeeze()
        ques_len = ques_len.view(-1, ques_len.size(0) * ques_len.size(1)).squeeze()
        option_len = option_len.view(-1, option_len.size(0) * option_len.size(1)).squeeze()

        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))

        outputs = self.bert(flat_input_ids, flat_token_type_ids, flat_attention_mask)
        sequence_output = outputs.last_hidden_state

        doc_ques_seq_output, ques_option_seq_output, doc_seq_output, ques_seq_output, option_seq_output = seperate_seq(
            sequence_output, doc_len, ques_len, option_len)

        pa_output = self.ssmatch([doc_seq_output, option_seq_output, option_len + 1])
        ap_output = self.ssmatch([option_seq_output, doc_seq_output, doc_len + 1])
        pq_output = self.ssmatch([doc_seq_output, ques_seq_output, ques_len + 1])
        qp_output = self.ssmatch([ques_seq_output, doc_seq_output, doc_len + 1])
        qa_output = self.ssmatch([ques_seq_output, option_seq_output, option_len + 1])
        aq_output = self.ssmatch([option_seq_output, ques_seq_output, ques_len + 1])

        pa_output_pool, _ = pa_output.max(1)
        ap_output_pool, _ = ap_output.max(1)
        pq_output_pool, _ = pq_output.max(1)
        qp_output_pool, _ = qp_output.max(1)
        qa_output_pool, _ = qa_output.max(1)
        aq_output_pool, _ = aq_output.max(1)

        pa_fuse = self.fuse([pa_output_pool, ap_output_pool])
        pq_fuse = self.fuse([pq_output_pool, qp_output_pool])
        qa_fuse = self.fuse([qa_output_pool, aq_output_pool])

        cat_pool = torch.cat([pa_fuse, pq_fuse, qa_fuse], 1)
        output_pool = self.dropout(cat_pool)
        match_logits = self.classifier3(output_pool)
        match_reshaped_logits = match_logits.view(-1, num_choices)

        match_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            match_loss = loss_fct(match_reshaped_logits, labels)

        if not return_dict:
            output = (match_reshaped_logits,) + outputs[2:]
            return ((match_loss,) + output) if match_loss is not None else output

        return MultipleChoiceModelOutput(
            loss=match_loss,
            logits=match_reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MeanPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# # DUMA
# class DUMA(nn.Module):
#     def __init__(self, config):
#         super(DUMA, self).__init__()
#         # self.map_linear = nn.Linear(2 * config.hidden_size, 2 * config.hidden_size)
#         # self.trans_linear = nn.Linear(config.hidden_size, config.hidden_size)
#         # self.drop_module = nn.Dropout(2 * config.hidden_dropout_prob)
#         # self.rank_module = nn.Linear(config.hidden_size * 2, 1)
#         self.doc_anttion = BertSelfAttention(config)
#         self.ques_opt_anttion = BertSelfAttention(config)
#         self.outputlayer = BertSelfOutput(config)
#
#         self.pooler = BertPooler(config)
#         # self.fuse = nn.Linear()
#
#     def forward(self, doc_seq_output, ques_option_seq_output ):
#         # proj_p, proj_q, seq_len = inputs
#         # trans_q = self.trans_linear(proj_q)
#         # att_weights = proj_p.bmm(torch.transpose(trans_q, 1, 2))
#         # att_norm = masked_softmax(att_weights, seq_len)
#         #
#         # att_vec = att_norm.bmm(proj_q)
#         # output = nn.ReLU()(self.trans_linear(att_vec))
#
#         doc_encoder = self.doc_anttion(doc_seq_output, encoder_hidden_states=ques_option_seq_output)
#         ques_option_encoder = self.ques_opt_anttion(ques_option_seq_output, encoder_hidden_states=doc_seq_output)
#         # fuse: summarize
#         # output = doc_encoder+ques_option_encoder
#         # output = torch.add(doc_encoder, ques_option_encoder)
#         output = self.outputlayer(doc_encoder[0], ques_option_encoder[0])
#
#         output = self.pooler(output)
#         return output

# # DUMA
# class DUMA(nn.Module):
#     def __init__(self, config):
#         super(DUMA, self).__init__()
#         self.doc_anttion = BertSelfAttention(config)
#         self.ques_opt_anttion = BertSelfAttention(config)
#         self.outputlayer = BertSelfOutput(config)
#
#         self.doc_pooler = BertPooler(config)
#         self.ques_opt_pooler = BertPooler(config)
#
#     def forward(self, sequence_output, doc_len, ques_len, option_len):
#         doc_ques_seq_output, ques_option_seq_output, doc_seq_output, ques_seq_output, option_seq_output = seperate_seq(
#             sequence_output, doc_len, ques_len, option_len)
#         doc_encoder = self.doc_anttion(doc_seq_output, encoder_hidden_states=ques_option_seq_output)
#         ques_option_encoder = self.ques_opt_anttion(ques_option_seq_output, encoder_hidden_states=doc_seq_output)
#         # fuse: summarize
#         # output = doc_encoder+ques_option_encoder
#         # output = torch.add(doc_encoder, ques_option_encoder)
#
#         doc_pooled_output = self.doc_pooler(doc_encoder[0])
#         ques_option_pooled_output = self.ques_opt_pooler(ques_option_encoder[0])
#
#         output = self.outputlayer(doc_pooled_output, ques_option_pooled_output)
#
#         output = self.pooler(output)
#         return output

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class DUMAOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cat = nn.Linear(2, 1)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = torch.cat([torch.unsqueeze(hidden_states, -1), torch.unsqueeze(input_tensor, -1)], axis=-1)
        hidden_states = self.cat(hidden_states)
        hidden_states = torch.squeeze(hidden_states, -1)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

# # DUMA
# class DUMA(nn.Module):
#     def __init__(self, config):
#         super(DUMA, self).__init__()
#         self.attention = BertSelfAttention(config)
#         self.pooler = MeanPooler(config)
#         self.outputlayer = DUMAOutput(config)
#
#     def forward(self, sequence_output, doc_len, ques_len, option_len, attention_mask=None):
#         doc_ques_seq_output, ques_option_seq_output, doc_seq_output, ques_seq_output, option_seq_output = seperate_seq(
#             sequence_output, doc_len, ques_len, option_len)
#         doc_encoder = self.attention(doc_seq_output, encoder_hidden_states=ques_option_seq_output,
#                                      attention_mask=attention_mask)
#         ques_option_encoder = self.attention(ques_option_seq_output, encoder_hidden_states=doc_seq_output,
#                                              attention_mask=attention_mask)
#         # fuse: summarize
#         # output = doc_encoder+ques_option_encoder
#         # output = torch.add(doc_encoder, ques_option_encoder)
#
#         doc_pooled_output = self.pooler(doc_encoder[0])
#         ques_option_pooled_output = self.pooler(ques_option_encoder[0])
#         # doc_pooled_output = mean_pooling(doc_encoder, attention_mask)
#         # ques_option_pooled_output = mean_pooling(ques_option_encoder, attention_mask)
#
#         output = self.outputlayer(doc_pooled_output, ques_option_pooled_output)
#
#         # output = self.pooler(output)
#         return output


# # DUMA
# class DUMA(nn.Module):
#     def __init__(self, config):
#         super(DUMA, self).__init__()
#         self.attention = BertSelfAttention(config)
#         self.pooler = MeanPooler(config)
#         self.outputlayer = BertSelfOutput(config)
#
#     def forward(self, sequence_output, doc_len, ques_len, option_len, attention_mask=None):
#         doc_ques_seq_output, ques_option_seq_output, doc_seq_output, ques_seq_output, option_seq_output = seperate_seq(
#             sequence_output, doc_len, ques_len, option_len)
#         doc_encoder = self.attention(doc_seq_output, encoder_hidden_states=ques_option_seq_output,
#                                      attention_mask=attention_mask)
#         ques_option_encoder = self.attention(ques_option_seq_output, encoder_hidden_states=doc_seq_output,
#                                              attention_mask=attention_mask)
#         # fuse: summarize
#         # output = doc_encoder+ques_option_encoder
#         # output = torch.add(doc_encoder, ques_option_encoder)
#         doc_pooled_output = self.pooler(doc_encoder[0])
#         ques_option_pooled_output = self.pooler(ques_option_encoder[0])
#
#         output = self.outputlayer(doc_pooled_output, ques_option_pooled_output)
#
#         output = self.pooler(output)
#         # output = mean_pooling(sequence_output, attention_mask)
#         return output

from model.DUMA_util import DUMA

class BertForMultipleChoiceWithDUMA(BertPreTrainedModel):
    def __init__(self, config, num_choices=2):
        super(BertForMultipleChoiceWithDUMA, self).__init__(config)
        self.num_choices = num_choices
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.duma = DUMA(config)
        duma = DUMA(config)
        self.dumas = nn.ModuleList([duma for _ in range(1)])
        self.pooler = self.bert.pooler
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, doc_len=None, ques_len=None, guid=None,
                option_len=None, labels=None, is_3=False, return_dict=None):
        r"""
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        doc_len = doc_len.view(-1, doc_len.size(0) * doc_len.size(1)).squeeze()
        ques_len = ques_len.view(-1, ques_len.size(0) * ques_len.size(1)).squeeze()
        option_len = option_len.view(-1, option_len.size(0) * option_len.size(1)).squeeze()

        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))

        outputs = self.bert(flat_input_ids, flat_token_type_ids, flat_attention_mask)
        sequence_output = outputs.last_hidden_state

        # doc_ques_seq_output, ques_option_seq_output, doc_seq_output, ques_seq_output, option_seq_output = seperate_seq(
        #     sequence_output, doc_len, ques_len, option_len)
        #
        # duma = self.duma(doc_seq_output, ques_option_seq_output)
        # duma = self.duma(sequence_output, doc_len, ques_len, option_len)
        # duma = self.duma(duma, doc_len, ques_len, option_len)
        for i, duma_module in enumerate(self.dumas):
            sequence_output = duma_module(sequence_output, doc_len, ques_len, option_len, flat_attention_mask)
        # pooled_output = self.pooler(sequence_output)
        pooled_output = mean_pooling(sequence_output, flat_attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        match_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            match_loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((match_loss,) + output) if match_loss is not None else output

        return MultipleChoiceModelOutput(
            loss=match_loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# class BertForMultipleChoiceWithDUMA(BertPreTrainedModel):
#     def __init__(self, config, num_choices=2):
#         super(BertForMultipleChoiceWithDUMA, self).__init__(config)
#         self.num_choices = num_choices
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         # self.duma = DUMA(config)
#         duma = DUMA(config)
#         self.dumas = nn.ModuleList([duma for _ in range(2)])
#         self.pooler = self.bert.pooler
#         self.classifier = nn.Linear(config.hidden_size, 1)
#         self.init_weights()
#
#     def forward(self, input_ids=None, token_type_ids=None, attention_mask=None,
#                 doc_len=None, ques_len=None, guid=None, option_len=None,
#                 position_ids=None, inputs_embeds=None,
#                 labels=None, is_3=False, return_dict=None):
#         r"""
#             labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
#             Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
#             num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
#             :obj:`input_ids` above)
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#         num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
#
#         flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
#         flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
#         flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
#         position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
#         inputs_embeds = (
#             inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
#             if inputs_embeds is not None
#             else None
#         )
#
#         # return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#         # num_choices = input_ids.shape[1]
#
#         # flat_input_ids = input_ids.view(-1, input_ids.size(-1))
#         doc_len = doc_len.view(-1, doc_len.size(0) * doc_len.size(1)).squeeze()
#         ques_len = ques_len.view(-1, ques_len.size(0) * ques_len.size(1)).squeeze()
#         option_len = option_len.view(-1, option_len.size(0) * option_len.size(1)).squeeze()
#
#         # flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
#         # flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
#
#         # outputs = self.bert(flat_input_ids, flat_token_type_ids, flat_attention_mask)
#         outputs = self.bert(
#             flat_input_ids,
#             attention_mask=flat_attention_mask,
#             token_type_ids=flat_token_type_ids,
#             position_ids=position_ids,
#             inputs_embeds=inputs_embeds,
#             return_dict=return_dict,
#         )
#
#         sequence_output = outputs.last_hidden_state
#
#         # doc_ques_seq_output, ques_option_seq_output, doc_seq_output, ques_seq_output, option_seq_output = seperate_seq(
#         #     sequence_output, doc_len, ques_len, option_len)
#         #
#         # duma = self.duma(doc_seq_output, ques_option_seq_output)
#         # duma = self.duma(sequence_output, doc_len, ques_len, option_len)
#         # duma = self.duma(duma, doc_len, ques_len, option_len)
#         for i, duma_module in enumerate(self.dumas):
#             sequence_output = duma_module(sequence_output, doc_len, ques_len, option_len, flat_attention_mask)
#         # pooled_output = self.pooler(sequence_output)
#         pooled_output = mean_pooling(sequence_output, flat_attention_mask)
#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)
#         reshaped_logits = logits.view(-1, num_choices)
#
#         match_loss = None
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             match_loss = loss_fct(reshaped_logits, labels)
#
#         if not return_dict:
#             output = (reshaped_logits,) + outputs[2:]
#             return ((match_loss,) + output) if match_loss is not None else output
#
#         return MultipleChoiceModelOutput(
#             loss=match_loss,
#             logits=reshaped_logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )


class BertForMultipleChoiceWithDUMAAndDCMN(BertPreTrainedModel):
    def __init__(self, config, num_choices=2):
        super(BertForMultipleChoiceWithDUMAAndDCMN, self).__init__(config)
        self.num_choices = num_choices
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dumas = nn.ModuleList([DUMA(config) for _ in range(2)])
        # self.pooler = self.bert.pooler
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size, 1)
        self.classifier2 = nn.Linear(2 * config.hidden_size, 1)
        self.classifier3 = nn.Linear(3 * config.hidden_size, 1)
        self.classifier4 = nn.Linear(4 * config.hidden_size, 1)
        self.classifier6 = nn.Linear(6 * config.hidden_size, 1)
        self.ssmatch = SSingleMatchNet(config)
        # self.pooler = BertPooler(config)
        self.fuse = FuseNet(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, doc_len=None, ques_len=None,
                position_ids=None, inputs_embeds=None,
                option_len=None, labels=None, is_3=False, return_dict=None):
        r"""
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # num_choices = input_ids.shape[1]

        # flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        doc_len = doc_len.view(-1, doc_len.size(0) * doc_len.size(1)).squeeze()
        ques_len = ques_len.view(-1, ques_len.size(0) * ques_len.size(1)).squeeze()
        option_len = option_len.view(-1, option_len.size(0) * option_len.size(1)).squeeze()

        # flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        # flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))

        # outputs = self.bert(flat_input_ids, flat_token_type_ids, flat_attention_mask)
        outputs = self.bert(
            flat_input_ids,
            attention_mask=flat_attention_mask,
            token_type_ids=flat_token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_dict=return_dict,
        )

        sequence_output = outputs.last_hidden_state

        # doc_ques_seq_output, ques_option_seq_output, doc_seq_output, ques_seq_output, option_seq_output = seperate_seq(
        #     sequence_output, doc_len, ques_len, option_len)
        #
        # duma = self.duma(doc_seq_output, ques_option_seq_output)
        # duma = self.duma(sequence_output, doc_len, ques_len, option_len)
        # duma = self.duma(duma, doc_len, ques_len, option_len)
        for i, duma_module in enumerate(self.dumas):
            sequence_output = duma_module(sequence_output, doc_len, ques_len, option_len, flat_attention_mask)
        # pooled_output = self.pooler(sequence_output)

        # dcmn start
        doc_ques_seq_output, ques_option_seq_output, doc_seq_output, ques_seq_output, option_seq_output = seperate_seq(
            sequence_output, doc_len, ques_len, option_len)

        pa_output = self.ssmatch([doc_seq_output, option_seq_output, option_len + 1])
        ap_output = self.ssmatch([option_seq_output, doc_seq_output, doc_len + 1])
        pq_output = self.ssmatch([doc_seq_output, ques_seq_output, ques_len + 1])
        qp_output = self.ssmatch([ques_seq_output, doc_seq_output, doc_len + 1])
        qa_output = self.ssmatch([ques_seq_output, option_seq_output, option_len + 1])
        aq_output = self.ssmatch([option_seq_output, ques_seq_output, ques_len + 1])

        pa_output_pool, _ = pa_output.max(1)
        ap_output_pool, _ = ap_output.max(1)
        pq_output_pool, _ = pq_output.max(1)
        qp_output_pool, _ = qp_output.max(1)
        qa_output_pool, _ = qa_output.max(1)
        aq_output_pool, _ = aq_output.max(1)

        pa_fuse = self.fuse([pa_output_pool, ap_output_pool])
        pq_fuse = self.fuse([pq_output_pool, qp_output_pool])
        qa_fuse = self.fuse([qa_output_pool, aq_output_pool])

        pooled_output = torch.cat([pa_fuse, pq_fuse, qa_fuse], 1)
        # dcmn end

        # pooled_output = mean_pooling(sequence_output, flat_attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier3(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        match_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            match_loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((match_loss,) + output) if match_loss is not None else output

        return MultipleChoiceModelOutput(
            loss=match_loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertForMultipleChoiceWithFocalLoss(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            # loss_fct = CrossEntropyLoss()
            loss_fct = FocalLoss(num_choices)
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FusionBertModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
    set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.rnn = nn.GRU(config.hidden_size, config.hidden_size, batch_first=True, bidirectional=True)
        self.linears = nn.ModuleList([nn.Linear(config.hidden_size * 2, 1) for i in range(12)])

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()

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
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        output_list = list()
        for i in range(len(encoder_outputs.all_hidden_states)):
            output, (final_hidden_state, final_cell_state) = self.rnn(encoder_outputs.all_hidden_states[i])
            output_list.append(self.linears[i](output))

        output_tensor = torch.cat(output_list, 2)
        output_tensor = F.softmax(output_tensor, 2)

        all_layers = torch.cat([torch.unsqueeze(i, 2) for i in encoder_outputs.all_hidden_states], axis=2)  # 第三维度拼接
        focus = torch.matmul(torch.unsqueeze(output_tensor, axis=2), all_layers)

        sequence_output = torch.squeeze(focus, 2)

        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class BertForMultipleChoiceWithFusion(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            # loss_fct = CrossEntropyLoss()
            loss_fct = FocalLoss(num_choices)
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def fusion_forward(self,
                   input_ids=None,
                   attention_mask=None,
                   token_type_ids=None,
                   position_ids=None,
                   head_mask=None,
                   inputs_embeds=None,
                   encoder_hidden_states=None,
                   encoder_attention_mask=None,
                   past_key_values=None,
                   use_cache=None,
                   output_attentions=None,
                   output_hidden_states=None,
                   return_dict=None, ):
    print('the forward is changed!')
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if self.config.is_decoder:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
    else:
        use_cache = False

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
        batch_size, seq_length = input_shape
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    device = input_ids.device if input_ids is not None else inputs_embeds.device

    # past_key_values_length
    past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

    if attention_mask is None:
        attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
    if token_type_ids is None:
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

    # If a 2D or 3D attention mask is provided for the cross-attention
    # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
    if self.config.is_decoder and encoder_hidden_states is not None:
        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
        encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
    else:
        encoder_extended_attention_mask = None

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x n_heads x N x N
    # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
    # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
    head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

    embedding_output = self.embeddings(
        input_ids=input_ids,
        position_ids=position_ids,
        token_type_ids=token_type_ids,
        inputs_embeds=inputs_embeds,
        past_key_values_length=past_key_values_length,
    )
    encoder_outputs = self.encoder(
        embedding_output,
        attention_mask=extended_attention_mask,
        head_mask=head_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_extended_attention_mask,
        past_key_values=past_key_values,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    sequence_output = concat_all_encoders_hidden_states(encoder_outputs.hidden_states, self.rnn, self.linears)
    # sequence_output = encoder_outputs[0]
    pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

    if not return_dict:
        return (sequence_output, pooled_output) + encoder_outputs[1:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        last_hidden_state=sequence_output,
        pooler_output=pooled_output,
        past_key_values=encoder_outputs.past_key_values,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
        cross_attentions=encoder_outputs.cross_attentions,
    )


def bind_fusion(model):
    # self.rnn = nn.GRU(config.hidden_size, config.hidden_size, batch_first=True, bidirectional=True)
    # self.linears = nn.ModuleList([nn.Linear(config.hidden_size * 2, 1) for i in range(12)])
    model.bert.rnn = nn.GRU(model.bert.config.hidden_size, model.bert.config.hidden_size, batch_first=True,
                            bidirectional=True)
    model.bert.linears = nn.ModuleList([nn.Linear(model.bert.config.hidden_size * 2, 1) for i in range(12)])

    from functools import partial
    forward = partial(fusion_forward, model.bert)
    model.bert.forward = forward
