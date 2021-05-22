import torch
from torch import nn

from base_model import BaseEncoderDecoder


class EncoderDecoder(BaseEncoderDecoder):
    """
    Encoder-Decoder model without Attention.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apply_attention(self, decoder_state, encoder_states,
                        mask_inference_inputs):
        """
        We do not apply attention here.
        """
        return decoder_state


class EncDecAttnDotProduct(BaseEncoderDecoder):
    """
    Encoder-Decoder models with Dot-Product Attention.
    """

    def __init__(self, mapping, bos, eos, embed_size=5,
                 enc_hidden_size=10, dec_hidden_size=10):
        # Since we apply dot-product
        assert enc_hidden_size == dec_hidden_size
        super().__init__(mapping, bos, eos, embed_size=embed_size,
                         enc_hidden_size=enc_hidden_size,
                         dec_hidden_size=dec_hidden_size)
        raise NotImplementedError

    def apply_attention(self, decoder_state, encoder_states,
                        mask_inference_inputs):
        raise NotImplementedError


class EncDecAttnBilinear(BaseEncoderDecoder):
    """
    Encoder-Decoder models with Bilinear Attention.
    """

    def __init__(self, mapping, bos, eos, embed_size=5,
                 enc_hidden_size=10, dec_hidden_size=10):
        assert enc_hidden_size == dec_hidden_size
        super().__init__(mapping, bos, eos, embed_size=embed_size,
                         enc_hidden_size=enc_hidden_size,
                         dec_hidden_size=dec_hidden_size)
        raise NotImplementedError

    def apply_attention(self, decoder_state, encoder_states,
                        mask_inference_inputs):
        raise NotImplementedError


class EncDecAttnConcat(BaseEncoderDecoder):
    """
    Encoder-Decoder models with Attention implemented by concatenation of
    decoder and encoder states and the subsequent linear layer application
    (+ tanh and dot-product).
    """

    def __init__(self, mapping, bos, eos, embed_size=5,
                 enc_hidden_size=10, dec_hidden_size=10):
        super().__init__(mapping, bos, eos, embed_size=embed_size,
                         enc_hidden_size=enc_hidden_size,
                         dec_hidden_size=dec_hidden_size)
        raise NotImplementedError

    def apply_attention(self, decoder_state, encoder_states,
                        mask_inference_inputs):
        raise NotImplementedError
