import numpy as np
import pandas as pd

import torch


def generate_sequence(vocabulary, seq_min_len=1, seq_max_len=10,
                      seq_count=100):
    """
    Generates sequences, each sequence contains values from the vocabulary.
    :param vocabulary: np.array with vocabulary values (bos and eos values are
    not included)
    :param seq_min_len: minimal length of generated sequence
    :param seq_max_len: maximal length of generated sequence
    :param seq_count: sequences count
    :return: generated sequence
    """
    sequences = []
    for i in range(seq_count):
        seq_size = np.random.randint(seq_min_len, seq_max_len + 1)
        sequence = np.random.choice(vocabulary, size=seq_size)
        sequences.append(sequence)
    return sequences


def generate_toy_dataset(vocabulary, seq_min_len=1, seq_max_len=10,
                         seq_count=100, reverse=False):
    """
    Generates toy copy dataset (or reverse copy dataset if `reverse` is True).
    :param vocabulary: np.array with vocabulary values (bos and eos values are
    not included)
    :param seq_min_len: minimal size of generated sequence
    :param seq_max_len: minimal size of generated sequence
    :param seq_count: sequences count
    :param reverse: if True reverse copy dataset will be returned
    :return: toy copy/reverse copy dataset
    """
    input_seqs = generate_sequence(
        vocabulary, seq_min_len=seq_min_len, seq_max_len=seq_max_len,
        seq_count=seq_count
    )
    output_seqs = input_seqs
    if reverse:
        output_seqs = []
        for sequence in input_seqs:
            output_seqs.append(sequence[::-1])
    data = {"input": input_seqs, "output": output_seqs}
    return pd.DataFrame(data)


def pad_sequence(sequence, seq_max_len, bos, eos):
    """
    Pad input sequence with begin of sequence item (`bos`) and end of sequence
    items (`eos`) to sequence with `seq_max_len` + 2 size (since we add bos and
    eos).
    :param sequence: np.array with sequence to be padded
    :param seq_max_len: maximal sequence size
    :param bos: begin of sequence item
    :param eos: end of sequence item
    :return: padded sequence with size seq_max_len + 2
    """
    padded_sequence = [bos]
    padded_sequence += list(sequence)
    for i in range(len(sequence) + 1, seq_max_len + 2):
        padded_sequence.append(eos)
    return np.array(padded_sequence)


def get_mask_inference(sequence, eos):
    """
    Form binary masks to be used for RNN batch inference and for loss
    proper computation. Since we padded input sequence we need to select just
    those states that do not belong to extra eos tokens (one last eos token we
    include).
    :param sequence: np.array with padded sequence
    :param eos: end of sequence item
    :return: torch.tensor with binary masks (seq_max_len)
    """
    mask = torch.zeros(sequence.shape[0], dtype=torch.bool)
    for i, item in enumerate(sequence):
        mask[i] = 1
        if item == eos:
            break
    return mask


def pad_sequences(dataset, seq_max_len, bos, eos):
    """
    Pad input and output sequences with bos and eos symbols to sequence with
    `seq_max_len` + 2 size and save it to `` and `` columns. Also build masks
    for loss computation
    :param dataset: toy dataset with columns `input` and `output`
    :param seq_max_len: maximal sequence length (without bos and eos symbols)
    :param bos: begin of sequence item
    :param eos: end of sequence item
    :return: dataset with
    """
    padded_input_seqs = [pad_sequence(input_seq, seq_max_len, bos, eos)
                         for input_seq in dataset["input"]]
    padded_output_seqs = [pad_sequence(output_seq, seq_max_len, bos, eos)
                          for output_seq in dataset["output"]]
    dataset["padded_input"] = padded_input_seqs
    dataset["padded_output"] = padded_output_seqs

    dataset["mask_inference_input"] = [
        get_mask_inference(input_seq, eos)
        for input_seq in dataset["padded_input"]
    ]
    return dataset


def one_hot_encode_sequence(sequence, mapping):
    """
    Encode the `sequence` based on `mapping`: str -> int.
    :param sequence: sequence of tokens
    :param mapping: mapping from tokens to index, dict(str: int)
    :return: one hot encoded sequence
    """
    one_hot = np.zeros((len(sequence), len(mapping)), dtype=np.int32)
    for i, item in enumerate(sequence):
        one_hot[i, mapping[item]] = 1
    return one_hot


def one_hot_encode_sequences(dataset, mapping):
    """
    Make One Hot encoding for sequences from dataset and save results in the
    dataset.
    :param dataset: pd.DataFrame with columns "padded_input" and "padded_output"
    :param mapping: mapping from tokens to index, dict(str: int)
    :return: modified dataset with new columns "one_hot_input" and
    "one_hot_output"
    """
    one_hot_input_seqs = [one_hot_encode_sequence(input_seq, mapping)
                          for input_seq in dataset["padded_input"]]
    one_hot_output_seqs = [one_hot_encode_sequence(output_seq, mapping)
                           for output_seq in dataset["padded_output"]]
    dataset["one_hot_input"] = one_hot_input_seqs
    dataset["one_hot_output"] = one_hot_output_seqs
    return dataset
