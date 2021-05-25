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

        # Softmax transforms coefficients only along `sequence` dimension
        self.softmax = nn.Softmax(dim=0)

    def apply_attention(self, decoder_state, encoder_states,
                        mask_inference_inputs):
        # decoder_state[:, :, None].shape = [1, batch_size, dec_hidden_size]
        weights = torch.mul(decoder_state[None, :, :], encoder_states)\
            .sum(dim=2)

        # [max_seq_len, batch_size]
        weights[torch.logical_not(mask_inference_inputs.t())] = -1e9
        weights = self.softmax.forward(weights)
        self.attn_weights.append(weights.t())  # we have to transpose because
        # weights.shape = [batch_size, max_seq_len] is expected in
        # `visualize_attention()`

        # Add encoder states with the calculated weights
        decoder_state = torch.mul(weights[:, :, None], encoder_states)\
            .sum(dim=0)
        return decoder_state


class EncDecAttnBilinear(BaseEncoderDecoder):
    """
    Encoder-Decoder models with Bilinear Attention.
    """

    def __init__(self, mapping, bos, eos, embed_size=5,
                 enc_hidden_size=10, dec_hidden_size=10):
        super().__init__(mapping, bos, eos, embed_size=embed_size,
                         enc_hidden_size=enc_hidden_size,
                         dec_hidden_size=dec_hidden_size)
        self.softmax = nn.Softmax(dim=0)
        self.attn_linear = nn.Linear(enc_hidden_size, dec_hidden_size)

    def apply_attention(self, decoder_state, encoder_states,
                        mask_inference_inputs):
        # decoder_state[:, :, None].shape = [1, batch_size, dec_hidden_size]
        weights = torch.mul(
            decoder_state[None, :, :], self.attn_linear(encoder_states)
        ).sum(dim=2)

        # [max_seq_len, batch_size]
        weights[torch.logical_not(mask_inference_inputs.t())] = -1e9
        weights = self.softmax.forward(weights)
        self.attn_weights.append(weights.t())  # we have to transpose because
        # weights.shape = [batch_size, max_seq_len] is expected in
        # `visualize_attention()`

        # Add encoder states with the calculated weights
        decoder_state = torch.mul(weights[:, :, None], encoder_states)\
            .sum(dim=0)
        return decoder_state


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
        self.softmax = nn.Softmax(dim=0)
        self.attn_linear = nn.Linear(enc_hidden_size + dec_hidden_size,
                                     dec_hidden_size)
        self.tanh = nn.Tanh()
        self.attn_vector = nn.Linear(dec_hidden_size, 1)

    def apply_attention(self, decoder_state, encoder_states,
                        mask_inference_inputs):
        # Concatenate along hidden_state dimensions
        seq_max_len = encoder_states.shape[0]
        # shape = [seq_max_len, batch_size, dec_hidden_size + enc_hidden_size]
        concat_states = torch.cat(
            (encoder_states,
             decoder_state[None, :, :].repeat((seq_max_len, 1, 1))),
            dim=2)
        # shape = [seq_max_len, batch_size, 1]
        weights = self.attn_vector(self.tanh(self.attn_linear(concat_states)))
        # [max_seq_len, batch_size]
        weights = weights.squeeze(2)
        weights[torch.logical_not(mask_inference_inputs.t())] = -1e9
        weights = self.softmax.forward(weights)
        self.attn_weights.append(weights.t())  # we have to transpose because
        # weights.shape = [batch_size, max_seq_len] is expected in
        # `visualize_attention()`

        # Add encoder states with the calculated weights
        decoder_state = torch.mul(weights[:, :, None], encoder_states)\
            .sum(dim=0)
        return decoder_state
