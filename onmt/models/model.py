""" Onmt NMT Model base class definition """
import torch.nn as nn
# from onmt.decoders.transformer import KGTransformerDecoder

class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, bptt=False, emotion=None, 
        src_concept_id=None, tgt_concept_emb=None, tgt_concept_words=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence of size ``(tgt_len, batch)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state
            emotion (LongTensor): The target emotions of size ``(batch,)``
            src_concept_id (LongTensor): The src concept ids of size ``(batch, 10)``
            tgt_concept_emb (FloatTensor): The tgt concept embedding of size ``(batch, emb_dim)``
            tgt_concept_words (tuple of (LongTensor, FloatTensor, FloatTensor)): The tgt concept words all in the shape of ``(batch, top_k)``

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        tgt = tgt[:-1]  # exclude last target from inputs

        enc_state, memory_bank, lengths = self.encoder(src, lengths)

        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)
        if hasattr(self.decoder, "tgt_concept_type"):
            dec_out, attns = self.decoder(tgt, memory_bank,
                                        memory_lengths=lengths,
                                        emotion=emotion,
                                        src_concept_id=src_concept_id,
                                        tgt_concept_emb=tgt_concept_emb,
                                        tgt_concept_words=tgt_concept_words)
        else:
            dec_out, attns = self.decoder(tgt, memory_bank,
                                        memory_lengths=lengths,
                                        emotion=emotion)
        return dec_out, attns

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)