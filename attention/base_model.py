import torch
from torch import nn

from abc import ABC, abstractmethod


class BaseEncoderDecoder(nn.Module, ABC):
    """
    Base Encoder-Decoder model.
    """

    def __init__(self, mapping, bos, eos, embed_size=5,
                 enc_hidden_size=10, dec_hidden_size=10):
        """
        :param mapping: mapping from token to index, dict(str: int)
        :param bos: begin of sequence item
        :param embed_size: embeddings vectors size
        :param enc_hidden_size: encoder hidden states size
        :param dec_hidden_size: decoder hidden states size
        """
        super().__init__()

        self.embed_size = embed_size
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.mapping = mapping
        self.bos = bos
        self.eos = eos

        self.embedding = nn.Linear(len(mapping), embed_size)
        self.encoder = nn.RNNCell(embed_size + len(mapping),
                                  self.enc_hidden_size)
        self.enc2dec_step = nn.Linear(self.enc_hidden_size,
                                      self.dec_hidden_size)
        self.decoder = nn.RNNCell(embed_size, self.dec_hidden_size)
        self.decoder_logits = nn.Linear(self.dec_hidden_size, len(mapping))
        self.log_softmax = nn.LogSoftmax(dim=1)
        
        # we store attention weights for visualizations
        self.attn_weights = []  # [seq_max_len + 1, batch_size, seq_max_len + 2]
        
    def clear_attn_weights(self):
        """
        Remove all attention weights.
        :return: None
        """
        self.attn_weights = []

    def encode(self, one_hot_inputs, mask_inference_inputs):
        """
        Encode input sequence.
        :param one_hot_inputs: torch.tensor with one hot encoded inputs with
        shape [batch_size, seq_max_len, len(mapping)]
        :param mask_inference_inputs: masks for RNN states inference selection.
        Since we padded input sequence we need to select just those states that
        do not belong to eos tokens.
        :return: encoder states for all input sequence, list of `seq_max_len`
        torch.tensors of shape (batch_size, enc_hidden_size)
        """
        batch_size, seq_max_len, voc_size = one_hot_inputs.shape

        # [batch_size, seq_max_len, enc_embed_size]
        embeddings = self.embedding(one_hot_inputs)

        position_matrix = torch.eye(seq_max_len, len(self.mapping))
        position_matrix = position_matrix.repeat(batch_size, 1, 1)
        embeddings = torch.cat((embeddings, position_matrix), dim=2)

        # The first state we set with 0 values
        state = torch.zeros((batch_size, self.enc_hidden_size),
                            dtype=torch.float32)
        states = []
        for i in range(seq_max_len):
            # [batch_size, hidden_size]
            next_state = self.encoder(embeddings[:, i], state)
            # save new state for not eos tokens, otherwise save prev state
            state = torch.where(
                torch.tile(mask_inference_inputs[:, i, None],
                           [1, next_state.shape[1]]),
                next_state, state
            )
            states.append(state)
        assert len(states) == seq_max_len
        return states

    def decode_init(self, encoder_states, mask_inference_inputs, eps=1e-20):
        """
        First decoder state and prediction extraction.
        :param encoder_states: encoder states for all input sequence, list of
        `seq_max_len` torch.tensors of shape (batch_size, enc_hidden_size)
        :param mask_inference_inputs: masks for RNN states inference selection.
        Since we padded input sequence we need to select just those states that
        do not belong to extra eos tokens (one last eos token we include).
        :param eps: eps parameter to be added to avoid log(0)
        :return: the first prediction and the first state of the decoder
        """
        last_encoder_state = encoder_states[-1]
        batch_size, _ = last_encoder_state.shape
        decoder_state = self.enc2dec_step(last_encoder_state)
        decoder_state = self.apply_attention(
            decoder_state, encoder_states, mask_inference_inputs)
        # For the first decoder state we always predict bos
        first_prediction = torch.zeros((batch_size, len(self.mapping)),
                                       dtype=torch.float32)
        first_prediction += eps  # we add `eps` to avoid log(0)
        first_prediction[:, 0] = 1.0
        return torch.log(first_prediction), decoder_state

    def decode_training(self, encoder_states, one_hot_outputs,
                        mask_inference_inputs):
        """
        Decode sequence for training mode.
        :param encoder_states: encoder states for all input sequence, list of
        `seq_max_len` torch.tensors of shape (batch_size, enc_hidden_size)
        :param one_hot_outputs: torch.tensor with one hot encoded outputs with
        shape [batch_size, seq_max_len, len(mapping)]
        :param mask_inference_inputs: masks for RNN states inference selection.
        Since we padded input sequence we need to select just those states that
        do not belong to eos tokens.
        :return: predictions
        """
        first_prediction, decoder_state = self.decode_init(
            encoder_states, mask_inference_inputs)
        predictions = [first_prediction]
        seq_max_len = one_hot_outputs.shape[1]
        for i in range(seq_max_len - 1):
            decoder_state, prediction = self.decode_step(
                decoder_state, one_hot_outputs[:, i], encoder_states,
                mask_inference_inputs
            )
            predictions.append(prediction)
        return torch.stack(predictions, dim=1)

    def decode_eval(self, encoder_states, seq_max_len, mask_inference_inputs,
                    eps=1e-20):
        """
        Decode sequence in evaluation mode.
        :param encoder_states: encoder states for all input sequence, list of
        `seq_max_len` torch.tensors of shape (batch_size, enc_hidden_size)
        :param seq_max_len: maximal sequence length
        :param mask_inference_inputs: masks for RNN states inference selection.
        Since we padded input sequence we need to select just those states that
        do not belong to eos tokens.
        :param eps: eps parameter to be added to avoid log(0)
        :return: argmax predictions, torch.tensor with shape (batch_size,
        seq_max_len + 2)
        """
        prediction, decoder_state = self.decode_init(encoder_states,
                                                     mask_inference_inputs)
        prediction = torch.argmax(prediction, dim=1)
        predictions = [prediction]
        batch_size = encoder_states[0].shape[0]
        for i in range(seq_max_len + 1):
            prev_prediction_oh = torch.zeros(batch_size, len(self.mapping),
                                             dtype=torch.float32) + eps
            prev_prediction_oh[:, prediction] = 1.0
            decoder_state, prediction = self.decode_step(
                decoder_state, prev_prediction_oh, encoder_states,
                mask_inference_inputs
            )
            prediction = torch.argmax(prediction, dim=1)
            predictions.append(prediction)
        return torch.stack(predictions, dim=1)

    @abstractmethod
    def apply_attention(self, decoder_state, encoder_states,
                        mask_inference_inputs):
        """
        You need to redefine this function in the inherited classes.
        This method applies attention for the current decoder state based on the
        given encoder states.
        :param decoder_state: current decoder state, torch.tensor with shape
        (batch_size, dec_hidden_size)
        :param encoder_states: list with the encoder states
        :param mask_inference_inputs: masks for RNN states inference selection.
        Since we padded input sequence we need to select just those states that
        do not belong to eos tokens.
        :return: updated decoder state
        """
        pass

    def decode_step(self, decoder_state, one_hot_outputs, encoder_states,
                    mask_inference_inputs):
        """
        Makes one decode step.
        :param decoder_state: previous decoder state
        :param one_hot_outputs: previous one hot encoded prediction
        :param encoder_states: encoder states for all input sequence, list of
        `saq_max_size` torch.tensors of shape (batch_size, enc_hidden_size)
        :param mask_inference_inputs: masks for RNN states inference selection.
        Since we padded input sequence we need to select just those states that
        do not belong to eos tokens.
        :return:
        """
        embeddings = self.embedding(one_hot_outputs)
        next_decoder_state = self.decoder(embeddings, decoder_state)
        next_decoder_state = self.apply_attention(
            next_decoder_state, encoder_states, mask_inference_inputs)
        prediction = self.decoder_logits(next_decoder_state)
        prediction = self.log_softmax(prediction)
        return next_decoder_state, prediction

    def forward(self, one_hot_inputs, mask_inference_inputs,
                one_hot_outputs=None, seq_max_len=None):
        """
        Main method to receive model predictions in training and evaluation
        modes.
        :param one_hot_inputs: torch.tensor with one hot encoded inputs with
        shape [batch_size, seq_max_len, len(mapping)]
        :param mask_inference_inputs: masks for RNN states inference selection.
        Since we padded input sequence we need to select just those states that
        do not belong to eos tokens.
        :param one_hot_outputs: torch.tensor with one hot encoded outputs with
        shape [batch_size, seq_max_len, len(mapping)]
        :param seq_max_len: maximal length of generated sequence
        :return: if self.training == True -> one hot encoded predicted output
        sequence with shape (batch_size, seq_max_len, len(mapping), otherwise ->
         argmax predictions, torch.tensor with shape (batch_size, seq_max_len
         + 2).
        """
        self.clear_attn_weights()
        if self.training is False:
            assert seq_max_len is not None
            with torch.no_grad():
                encoder_states = self.encode(one_hot_inputs,
                                             mask_inference_inputs)
                return self.decode_eval(encoder_states, seq_max_len,
                                        mask_inference_inputs)

        assert one_hot_outputs is not None
        encoder_states = self.encode(one_hot_inputs, mask_inference_inputs)
        return self.decode_training(encoder_states, one_hot_outputs,
                                    mask_inference_inputs)

    def apply_mapping(self, predictions):
        """
        Applies mapping and remove bos and eos from predicted sequences.
        :param predictions: argmax predictions, torch.tensor with shape
        (batch_size, seq_max_len + 2).
        :return: list with restored sequences from the network prediction,
        [batch_size, predicted_seq_len]
        """
        num2sym = {self.mapping[sym]: sym for sym in self.mapping}
        results = []
        for prediction in predictions.numpy():
            result = []
            for elem in prediction:
                if elem == self.mapping[self.bos]:
                    continue
                if elem == self.mapping[self.eos]:
                    break
                result.append(num2sym[elem])
            results.append(result)
        return results
