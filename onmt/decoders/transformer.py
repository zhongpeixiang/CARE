"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn

from onmt.decoders.decoder import DecoderBase
from onmt.modules import MultiHeadedAttention, AverageAttention
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.utils.misc import sequence_mask


class TransformerDecoderLayer(nn.Module):
    """
    Args:
      d_model (int): the dimension of keys/values/queries in
          :class:`MultiHeadedAttention`, also the input size of
          the first-layer of the :class:`PositionwiseFeedForward`.
      heads (int): the number of heads for MultiHeadedAttention.
      d_ff (int): the second-layer of the :class:`PositionwiseFeedForward`.
      dropout (float): dropout probability.
      self_attn_type (string): type of self-attention scaled-dot, average
    """

    def __init__(self, d_model, heads, d_ff, dropout, attention_dropout,
                 self_attn_type="scaled-dot", max_relative_positions=0,
                 aan_useffn=False, tgt_concept_words_type=-1):
        super(TransformerDecoderLayer, self).__init__()

        if self_attn_type == "scaled-dot":
            self.self_attn = MultiHeadedAttention(
                heads, d_model, dropout=dropout,
                max_relative_positions=max_relative_positions)
        elif self_attn_type == "average":
            self.self_attn = AverageAttention(d_model,
                                              dropout=attention_dropout,
                                              aan_useffn=aan_useffn)

        self.context_attn = MultiHeadedAttention(
            heads, d_model, dropout=attention_dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)
        
        self.tgt_concept_words_type = tgt_concept_words_type
        if tgt_concept_words_type in [2]:
            self.tgt_concept_mlp = nn.Linear(d_model*2, d_model)

    def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask,
                layer_cache=None, step=None, tgt_concept_words_emb=None, tgt_concept_words_type=-1):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, 1, model_dim)``
            memory_bank (FloatTensor): ``(batch_size, src_len, model_dim)``
            src_pad_mask (LongTensor): ``(batch_size, 1, src_len)``
            tgt_pad_mask (LongTensor): ``(batch_size, 1, 1)``

        Returns:
            (FloatTensor, FloatTensor):

            * output ``(batch_size, 1, model_dim)``
            * attn ``(batch_size, 1, src_len)``

        """
        dec_mask = None
        if step is None:
            tgt_len = tgt_pad_mask.size(-1)
            future_mask = torch.ones(
                [tgt_len, tgt_len],
                device=tgt_pad_mask.device,
                dtype=torch.uint8)
            future_mask = future_mask.triu_(1).view(1, tgt_len, tgt_len)
            # BoolTensor was introduced in pytorch 1.2
            try:
                future_mask = future_mask.bool()
            except AttributeError:
                pass
            dec_mask = torch.gt(tgt_pad_mask + future_mask, 0)

        input_norm = self.layer_norm_1(inputs)

        if isinstance(self.self_attn, MultiHeadedAttention):
            query, attn = self.self_attn(input_norm, input_norm, input_norm,
                                         mask=dec_mask,
                                         layer_cache=layer_cache,
                                         attn_type="self")
        elif isinstance(self.self_attn, AverageAttention):
            query, attn = self.self_attn(input_norm, mask=dec_mask,
                                         layer_cache=layer_cache, step=step)

        query = self.drop(query) + inputs

        # ablation
        if tgt_concept_words_emb is not None:
            # print(query.shape, tgt_concept_words_emb.shape)
            if self.tgt_concept_words_type == 2:
                query = self.tgt_concept_mlp(torch.cat([query, tgt_concept_words_emb], dim=2))
            if self.tgt_concept_words_type == 3:
                query = (query + tgt_concept_words_emb)/2

        query_norm = self.layer_norm_2(query)
        mid, attn = self.context_attn(memory_bank, memory_bank, query_norm,
                                      mask=src_pad_mask,
                                      layer_cache=layer_cache,
                                      attn_type="context")
        output = self.feed_forward(self.drop(mid) + query)

        return output, attn

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.context_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.drop.p = dropout


class TransformerDecoder(DecoderBase):
    """The Transformer decoder from "Attention is All You Need".
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          BB[multi-head src-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> BB
          BB --> C
          C --> O


    Args:
       num_layers (int): number of encoder layers.
       d_model (int): size of the model
       heads (int): number of heads
       d_ff (int): size of the inner FF layer
       copy_attn (bool): if using a separate copy attention
       self_attn_type (str): type of self-attention scaled-dot, average
       dropout (float): dropout parameters
       embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings
    """

    def __init__(self, num_layers, d_model, heads, d_ff,
                 copy_attn, self_attn_type, dropout, attention_dropout,
                 embeddings, max_relative_positions, aan_useffn,
                 num_emotion_classes=0, emotion_emb_size=0):
        super(TransformerDecoder, self).__init__()

        self.embeddings = embeddings

        # Emotion embedding
        self.num_emotion_classes = num_emotion_classes
        self.emotion_emb_size = emotion_emb_size
        if num_emotion_classes != 0 and emotion_emb_size != 0:
            self.emo_embedding = nn.Embedding(num_emotion_classes, emotion_emb_size)
            self.emo_mlp = nn.Linear(d_model+emotion_emb_size, d_model)

        # Decoder State
        self.state = {}

        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, heads, d_ff, dropout,
             attention_dropout, self_attn_type=self_attn_type,
             max_relative_positions=max_relative_positions,
             aan_useffn=aan_useffn)
             for i in range(num_layers)])

        # previously, there was a GlobalAttention module here for copy
        # attention. But it was never actually used -- the "copy" attention
        # just reuses the context attention.
        self._copy = copy_attn
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.dec_layers,
            opt.dec_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.copy_attn,
            opt.self_attn_type,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.attention_dropout[0] if type(opt.attention_dropout)
            is list else opt.dropout,
            embeddings,
            opt.max_relative_positions,
            opt.aan_useffn,
            opt.num_emotion_classes,
            opt.emotion_emb_size)

    def init_state(self, src, memory_bank, enc_hidden):
        """Initialize decoder state."""
        self.state["src"] = src
        self.state["cache"] = None

    def map_state(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        self.state["src"] = fn(self.state["src"], 1)
        if self.state["cache"] is not None:
            _recursive_map(self.state["cache"])

    def detach_state(self):
        self.state["src"] = self.state["src"].detach()

    def forward(self, tgt, memory_bank, step=None, emotion=None, **kwargs):
        """Decode, possibly stepwise."""
        if step == 0:
            self._init_cache(memory_bank)

        tgt_words = tgt[:, :, 0].transpose(0, 1)

        emb = self.embeddings(tgt, step=step)
        assert emb.dim() == 3  # len x batch x embedding_dim

        # add emotion embedding using linear transformation
        if emotion is not None:
            batch_emotion_embedding = self.emo_embedding(emotion) # (bacth, emotion_emb_size)
            batch_emotion_embedding = batch_emotion_embedding.unsqueeze(0).repeat(emb.size(0), 1, 1) # (len, bacth, emotion_emb_size)
            emb = self.emo_mlp(torch.cat([emb, batch_emotion_embedding], dim=2)) # emb: (len, bacth, embedding_dim)
        
        output = emb.transpose(0, 1).contiguous()
        src_memory_bank = memory_bank.transpose(0, 1).contiguous()

        pad_idx = self.embeddings.word_padding_idx
        src_lens = kwargs["memory_lengths"]
        src_max_len = self.state["src"].shape[0]
        src_pad_mask = ~sequence_mask(src_lens, src_max_len).unsqueeze(1)
        tgt_pad_mask = tgt_words.data.eq(pad_idx).unsqueeze(1)  # [B, 1, T_tgt]

        for i, layer in enumerate(self.transformer_layers):
            layer_cache = self.state["cache"]["layer_{}".format(i)] \
                if step is not None else None
            output, attn = layer(
                output,
                src_memory_bank,
                src_pad_mask,
                tgt_pad_mask,
                layer_cache=layer_cache,
                step=step)

        output = self.layer_norm(output)
        dec_outs = output.transpose(0, 1).contiguous()
        attn = attn.transpose(0, 1).contiguous()

        attns = {"std": attn}
        if self._copy:
            attns["copy"] = attn

        # TODO change the way attns is returned dict => list or tuple (onnx)
        return dec_outs, attns

    def _init_cache(self, memory_bank):
        self.state["cache"] = {}
        batch_size = memory_bank.size(1)
        depth = memory_bank.size(-1)

        for i, layer in enumerate(self.transformer_layers):
            layer_cache = {"memory_keys": None, "memory_values": None}
            if isinstance(layer.self_attn, AverageAttention):
                layer_cache["prev_g"] = torch.zeros((batch_size, 1, depth),
                                                    device=memory_bank.device)
            else:
                layer_cache["self_keys"] = None
                layer_cache["self_values"] = None
            self.state["cache"]["layer_{}".format(i)] = layer_cache

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer_layers:
            layer.update_dropout(dropout, attention_dropout)


class KGTransformerDecoder(DecoderBase):
    """The Transformer decoder from "Attention is All You Need".
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          BB[multi-head src-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> BB
          BB --> C
          C --> O


    Args:
       num_layers (int): number of encoder layers.
       d_model (int): size of the model
       heads (int): number of heads
       d_ff (int): size of the inner FF layer
       copy_attn (bool): if using a separate copy attention
       self_attn_type (str): type of self-attention scaled-dot, average
       dropout (float): dropout parameters
       embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings
    """

    def __init__(self, num_layers, d_model, heads, d_ff,
                 copy_attn, self_attn_type, dropout, attention_dropout,
                 embeddings, max_relative_positions, aan_useffn,
                 num_emotion_classes=0, emotion_emb_size=0, 
                 train_tgt_concept_embedding="", tgt_concept_type=-1, 
                 train_tgt_concept_words="", tgt_concept_words_type=-1):
        super(KGTransformerDecoder, self).__init__()

        self.embeddings = embeddings
        self.vocab_size = self.embeddings.word_vocab_size
        self.tgt_concept_type = tgt_concept_type
        self.train_tgt_concept_words = train_tgt_concept_words
        self.tgt_concept_words_type = tgt_concept_words_type

        # Emotion embedding
        self.num_emotion_classes = num_emotion_classes
        self.emotion_emb_size = emotion_emb_size
        if num_emotion_classes != 0 and tgt_concept_type in [0,1,2,3]:
            self.emo_embedding = nn.Embedding(num_emotion_classes, emotion_emb_size)
            if self.tgt_concept_type == 0:
                self.tgt_concept_mlp = nn.Linear(d_model*2+emotion_emb_size, d_model)
            if self.tgt_concept_type == 1:
                self.emo_mlp = nn.Linear(d_model+emotion_emb_size, d_model)
            if self.tgt_concept_type in [2,3]:
                self.emo_mlp = nn.Linear(d_model+emotion_emb_size, d_model)
                self.tgt_concept_mlp = nn.Linear(d_model*2, d_model)

        if num_emotion_classes == 0 and self.tgt_concept_type in [0,2,3]:
            self.tgt_concept_mlp = nn.Linear(d_model*2, d_model)
        
        # tgt concept words
        if self.tgt_concept_words_type in [0,1,2,3]:
            self.score_temp = nn.Parameter(torch.ones(self.vocab_size))
            self.VAD_score_temp = nn.Parameter(torch.ones(self.vocab_size))
            self.src_relevance_temp = nn.Parameter(torch.ones(self.vocab_size))
            self.VAD_lambda = nn.Parameter(torch.ones(self.vocab_size)*0.5)

        if num_emotion_classes != 0 and self.tgt_concept_words_type in [0,1,2,3]:
            self.emo_embedding = nn.Embedding(num_emotion_classes, emotion_emb_size)
            if self.tgt_concept_words_type == 0:
                self.tgt_concept_mlp = nn.Linear(d_model*2+emotion_emb_size, d_model)
            if self.tgt_concept_words_type == 1:
                self.emo_mlp = nn.Linear(d_model+emotion_emb_size, d_model)
            if self.tgt_concept_words_type == 2:
                self.emo_mlp = nn.Linear(d_model+emotion_emb_size, d_model)
                # self.tgt_concept_mlp = nn.Linear(d_model*2, d_model)
            if self.tgt_concept_words_type == 3:
                self.emo_mlp = nn.Linear(d_model+emotion_emb_size, d_model)

        if num_emotion_classes == 0 and self.tgt_concept_words_type in [0]:
            self.tgt_concept_mlp = nn.Linear(d_model*2, d_model)
        

        # if self.train_tgt_concept_words != "":
        #     self.score_temp = nn.Parameter(torch.ones(self.vocab_size))
        #     self.VAD_score_temp = nn.Parameter(torch.ones(self.vocab_size))
        #     self.VAD_lambda = nn.Parameter(torch.ones(self.vocab_size)*0.5)

        #     if num_emotion_classes != 0 and self.tgt_concept_type == 0:
        #         self.emo_embedding = nn.Embedding(num_emotion_classes, emotion_emb_size)
        #         self.tgt_concept_mlp = nn.Linear(d_model*2+emotion_emb_size, d_model)
        #     else:
        #         self.tgt_concept_mlp = nn.Linear(d_model*2, d_model)
        
        
        # if num_emotion_classes != 0 and emotion_emb_size != 0 and train_tgt_concept_embedding and if self.tgt_concept_type in [0]:
        #     self.emo_embedding = nn.Embedding(num_emotion_classes, emotion_emb_size)
        #     self.emo_mlp = nn.Linear(2*d_model+emotion_emb_size, d_model)
        # if num_emotion_classes != 0 and emotion_emb_size != 0 and train_tgt_concept_embedding == "":
        #     self.emo_embedding = nn.Embedding(num_emotion_classes, emotion_emb_size)
        #     self.emo_mlp = nn.Linear(d_model+emotion_emb_size, d_model)
        # if num_emotion_classes == 0 and train_tgt_concept_embedding:
        #     if self.tgt_concept_type in [0,2,3]:
        #         self.tgt_concept_mlp = nn.Linear(d_model*2, d_model)
        
        # Decoder State
        self.state = {}

        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, heads, d_ff, dropout,
             attention_dropout, self_attn_type=self_attn_type,
             max_relative_positions=max_relative_positions,
             aan_useffn=aan_useffn, tgt_concept_words_type=tgt_concept_words_type)
             for i in range(num_layers)])

        # previously, there was a GlobalAttention module here for copy
        # attention. But it was never actually used -- the "copy" attention
        # just reuses the context attention.
        self._copy = copy_attn
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.dec_layers,
            opt.dec_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.copy_attn,
            opt.self_attn_type,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.attention_dropout[0] if type(opt.attention_dropout)
            is list else opt.dropout,
            embeddings,
            opt.max_relative_positions,
            opt.aan_useffn,
            opt.num_emotion_classes,
            opt.emotion_emb_size,
            opt.train_tgt_concept_embedding,
            opt.tgt_concept_type,
            opt.train_tgt_concept_words,
            opt.tgt_concept_words_type)

    def init_state(self, src, memory_bank, enc_hidden):
        """Initialize decoder state."""
        self.state["src"] = src
        self.state["cache"] = None

    def map_state(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        self.state["src"] = fn(self.state["src"], 1)
        if self.state["cache"] is not None:
            _recursive_map(self.state["cache"])

    def detach_state(self):
        self.state["src"] = self.state["src"].detach()

    def forward(self, tgt, memory_bank, step=None, emotion=None, 
        src_concept_id=None, tgt_concept_emb=None, tgt_concept_words=None, **kwargs):
        """Decode, possibly stepwise."""
        """
            memory_bank: ``(src_len, batch_size, model_dim)``
        """
        if step == 0:
            self._init_cache(memory_bank)

        tgt_words = tgt[:, :, 0].transpose(0, 1)

        emb = self.embeddings(tgt, step=step)
        assert emb.dim() == 3  # len x batch x embedding_dim

        # if emotion is not None:
        #     batch_emotion_embedding = self.emo_embedding(emotion) # (bacth, emotion_emb_size)
        #     batch_emotion_embedding = batch_emotion_embedding.unsqueeze(0).repeat(emb.size(0), 1, 1) # (len, bacth, emotion_emb_size)
        #     emb = torch.cat([emb, batch_emotion_embedding], dim=2)
        # if tgt_concept_emb is not None and self.tgt_concept_type == 0:
        #     tgt_concept_emb = tgt_concept_emb.unsqueeze(0).repeat(emb.size(0), 1, 1) # (len, bacth, d_model)
        #     emb = torch.cat([emb, tgt_concept_emb], dim=2)
        # if tgt_concept_emb is not None and self.tgt_concept_type == 1:
        #     tgt_concept_emb = tgt_concept_emb.unsqueeze(0).repeat(emb.size(0), 1, 1) # (len, bacth, d_model)
        #     emb = (emb + tgt_concept_emb)/2
        
        # add emotion embedding using linear transformation
        # if emotion is not None and tgt_concept_emb is not None:
        #     if self.tgt_concept_type == 0:
        #         batch_emotion_embedding = self.emo_embedding(emotion) # (bacth, emotion_emb_size)
        #         batch_emotion_embedding = batch_emotion_embedding.unsqueeze(0).repeat(emb.size(0), 1, 1) # (len, bacth, emotion_emb_size)
        #         tgt_concept_emb = tgt_concept_emb.unsqueeze(0).repeat(emb.size(0), 1, 1) # (len, bacth, d_model)
        #         emb = self.tgt_concept_mlp(torch.cat([emb, batch_emotion_embedding, tgt_concept_emb], dim=2)) # emb: (len, bacth, d_model)
        #     if self.tgt_concept_type == 1:
        #         batch_emotion_embedding = self.emo_embedding(emotion) # (bacth, emotion_emb_size)
        #         batch_emotion_embedding = batch_emotion_embedding.unsqueeze(0).repeat(emb.size(0), 1, 1) # (len, bacth, emotion_emb_size)
        #         tgt_concept_emb = tgt_concept_emb.unsqueeze(0).repeat(emb.size(0), 1, 1) # (len, bacth, d_model)
        #         emb = self.emo_mlp(torch.cat([emb, batch_emotion_embedding], dim=2)) # emb: (len, bacth, d_model)
        #         emb = (emb + tgt_concept_emb)/2
        #     if self.tgt_concept_type in [2,3]:
        #         batch_emotion_embedding = self.emo_embedding(emotion) # (bacth, emotion_emb_size)
        #         batch_emotion_embedding = batch_emotion_embedding.unsqueeze(0).repeat(emb.size(0), 1, 1) # (len, bacth, emotion_emb_size)
        #         emb = self.emo_mlp(torch.cat([emb, batch_emotion_embedding], dim=2)) # emb: (len, bacth, d_model)
        # if emotion is not None and tgt_concept_emb is None:
        #     batch_emotion_embedding = self.emo_embedding(emotion) # (bacth, emotion_emb_size)
        #     batch_emotion_embedding = batch_emotion_embedding.unsqueeze(0).repeat(emb.size(0), 1, 1) # (len, bacth, emotion_emb_size)
        #     emb = self.emo_mlp(torch.cat([emb, batch_emotion_embedding], dim=2)) # emb: (len, bacth, d_model)
        # if emotion is None and tgt_concept_emb is not None:
        #     if self.tgt_concept_type == 0:
        #         tgt_concept_emb = tgt_concept_emb.unsqueeze(0).repeat(emb.size(0), 1, 1) # (len, bacth, d_model)
        #         emb = self.tgt_concept_mlp(torch.cat([emb, tgt_concept_emb], dim=2)) # emb: (len, bacth, d_model)
        #     if self.tgt_concept_type == 1:
        #         tgt_concept_emb = tgt_concept_emb.unsqueeze(0).repeat(emb.size(0), 1, 1) # (len, bacth, d_model)
        #         emb = (emb + tgt_concept_emb)/2
        #     if self.tgt_concept_type in [2,3]:
        #         pass
        tgt_concept_words_emb = None
        if tgt_concept_words is not None:
            score_softmax = torch.softmax(self.score_temp[tgt_concept_words[0]] * tgt_concept_words[1], dim=-1) # (batch, top_k)
            VAD_score_softmax = torch.softmax(self.VAD_score_temp[tgt_concept_words[0]] * tgt_concept_words[2], dim=-1) # (batch, top_k)
            tgt_concept_words_emb = self.embeddings.word_lut(tgt_concept_words[0]) # (batch, top_k, d_model)

            # src relevance
            # src_relevance = torch.softmax(self.src_relevance_temp[tgt_concept_words[0]] * torch.bmm(tgt_concept_words_emb, memory_bank.mean(dim=0).unsqueeze(-1)).squeeze(-1), dim=-1) # (batch, top_k)
            # score_softmax = (score_softmax + src_relevance)/2

            clamped_VAD_lambda = self.VAD_lambda.clamp(1e-4, 1-(1e-4))
            weight = score_softmax * clamped_VAD_lambda[tgt_concept_words[0]] + VAD_score_softmax * (1-clamped_VAD_lambda[tgt_concept_words[0]])
            tgt_concept_words_emb = (weight.unsqueeze(-1) * tgt_concept_words_emb).sum(dim=1) # (batch, d_model)
            tgt_concept_words_emb = tgt_concept_words_emb.unsqueeze(0).repeat(emb.size(0), 1, 1) # (len, bacth, d_model)
            if emotion is not None and self.tgt_concept_words_type in [0]:
                batch_emotion_embedding = self.emo_embedding(emotion) # (bacth, emotion_emb_size)
                batch_emotion_embedding = batch_emotion_embedding.unsqueeze(0).repeat(emb.size(0), 1, 1) # (len, bacth, emotion_emb_size)
                emb = self.tgt_concept_mlp(torch.cat([emb, tgt_concept_words_emb, batch_emotion_embedding], dim=2)) # emb: (len, bacth, d_model)
            if emotion is None and self.tgt_concept_words_type in [0]:
                emb = self.tgt_concept_mlp(torch.cat([emb, tgt_concept_words_emb], dim=2)) # emb: (len, bacth, d_model)
            
            if emotion is not None and self.tgt_concept_words_type in [1]:
                batch_emotion_embedding = self.emo_embedding(emotion) # (bacth, emotion_emb_size)
                batch_emotion_embedding = batch_emotion_embedding.unsqueeze(0).repeat(emb.size(0), 1, 1) # (len, bacth, emotion_emb_size)
                emb = self.emo_mlp(torch.cat([emb, batch_emotion_embedding], dim=2)) # emb: (len, bacth, d_model)
                emb = (emb + tgt_concept_words_emb)/2
            if emotion is None and self.tgt_concept_words_type in [1]:
                emb = (emb + tgt_concept_words_emb)/2
            
            if emotion is not None and self.tgt_concept_words_type in [2,3]:
                batch_emotion_embedding = self.emo_embedding(emotion) # (bacth, emotion_emb_size)
                batch_emotion_embedding = batch_emotion_embedding.unsqueeze(0).repeat(emb.size(0), 1, 1) # (len, bacth, emotion_emb_size)
                emb = self.emo_mlp(torch.cat([emb, batch_emotion_embedding], dim=2)) # emb: (len, bacth, d_model)
            if emotion is None and self.tgt_concept_words_type in [2,3]:
                pass

        output = emb.transpose(0, 1).contiguous()
        src_memory_bank = memory_bank.transpose(0, 1).contiguous()

        pad_idx = self.embeddings.word_padding_idx
        src_lens = kwargs["memory_lengths"]
        src_max_len = self.state["src"].shape[0]
        src_pad_mask = ~sequence_mask(src_lens, src_max_len).unsqueeze(1)
        tgt_pad_mask = tgt_words.data.eq(pad_idx).unsqueeze(1)  # [B, 1, T_tgt]


        for i, layer in enumerate(self.transformer_layers):
            layer_cache = self.state["cache"]["layer_{}".format(i)] \
                if step is not None else None
            output, attn = layer(
                output,
                src_memory_bank,
                src_pad_mask,
                tgt_pad_mask,
                layer_cache=layer_cache,
                step=step,
                tgt_concept_words_emb=tgt_concept_words_emb.transpose(1,0))

        # if tgt_concept_emb is not None:
        #     if self.tgt_concept_type == 2:
        #         tgt_concept_emb = tgt_concept_emb.unsqueeze(1).repeat(1, output.size(1), 1) # (bacth, len, emotion_emb_size)
        #         output = self.tgt_concept_mlp(torch.cat([output, tgt_concept_emb], dim=2)) # (bacth, len, embedding_dim)
        
        output = self.layer_norm(output)
        dec_outs = output.transpose(0, 1).contiguous()
        attn = attn.transpose(0, 1).contiguous()

        attns = {"std": attn}
        if self._copy:
            attns["copy"] = attn

        # TODO change the way attns is returned dict => list or tuple (onnx)
        # if tgt_concept_emb is not None:
        #     if self.tgt_concept_type == 3:
        #         tgt_concept_emb = tgt_concept_emb.unsqueeze(0).repeat(dec_outs.size(0), 1, 1) # (len, bacth, emotion_emb_size)
        #         dec_outs = self.tgt_concept_mlp(torch.cat([dec_outs, tgt_concept_emb], dim=2)) # emb: (len, bacth, embedding_dim)
        
        return dec_outs, attns

    def _init_cache(self, memory_bank):
        self.state["cache"] = {}
        batch_size = memory_bank.size(1)
        depth = memory_bank.size(-1)

        for i, layer in enumerate(self.transformer_layers):
            layer_cache = {"memory_keys": None, "memory_values": None}
            if isinstance(layer.self_attn, AverageAttention):
                layer_cache["prev_g"] = torch.zeros((batch_size, 1, depth),
                                                    device=memory_bank.device)
            else:
                layer_cache["self_keys"] = None
                layer_cache["self_values"] = None
            self.state["cache"]["layer_{}".format(i)] = layer_cache

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer_layers:
            layer.update_dropout(dropout, attention_dropout)