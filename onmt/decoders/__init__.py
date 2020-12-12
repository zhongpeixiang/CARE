"""Module defining decoders."""
from onmt.decoders.decoder import DecoderBase, InputFeedRNNDecoder, \
    StdRNNDecoder, EDSDecoder
from onmt.decoders.transformer import TransformerDecoder, KGTransformerDecoder
from onmt.decoders.cnn_decoder import CNNDecoder


str2dec = {"rnn": StdRNNDecoder, "ifrnn": InputFeedRNNDecoder,
           "cnn": CNNDecoder, "transformer": TransformerDecoder, 
           "kg_transformer": KGTransformerDecoder, "eds": EDSDecoder}

__all__ = ["DecoderBase", "TransformerDecoder", "StdRNNDecoder", "CNNDecoder",
           "InputFeedRNNDecoder", "str2dec"]
