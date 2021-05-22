import numpy as np
import torch

from IPython import display  # для красивого отображения графиков
import matplotlib.pyplot as plt

from Levenshtein import distance as levenshtein_distance  # для оценки качества


def format_data(series):
    """
    Modify data to the format is acceptable for the neural network.
    :param series: pd.Series with data to be formatted
    :return: data acceptable for the neural network
    """
    return torch.from_numpy(np.stack(series.values))


def batch_generator(dataset, batch_size, shuffle=True, return_source=False):
    """
    Generate batches for training.
    :param dataset: dataset with "one_hot_input", "mask_inference_input" and
    "one_hot_output" columns
    :param batch_size: batch size
    :param shuffle: if True shuffles the data
    :param return_source: if True additionally return "input" and "output"
    columns values
    :return: input, masks for input, output sequences for the batch
    """
    indices = np.arange(len(dataset))
    if shuffle:
        indices = np.random.permutation(indices)

    for start in range(0, len(indices), batch_size):
        end = start + batch_size
        if end > len(indices):
            end = len(indices)
        selected = dataset.iloc[start:end]
        output = [
            format_data(selected["one_hot_input"]).float(),
            format_data(selected["mask_inference_input"]),
            format_data(selected["one_hot_output"]).float()
        ]
        if not return_source:
            yield tuple(output)
        else:
            output += [selected["input"].values, selected["output"].values]
            yield tuple(output)


def calculate_loss(prediction, mask_inference_input,
                   one_hot_output):
    """
    Calculates loss for padded sequences based on binary masks.
    :param prediction: batch predictions for which log_softmax was applied,
    (batch_size, seq_max_len, len(sym2num))
    :param mask_inference_input: input binary masks
    :param one_hot_output: one hot encoded ground truth outputs
    :return: loss values, torch.tensor with shape (batch_size,)
    """
    return - torch.sum(prediction * one_hot_output *
                       mask_inference_input[:, :, None]) \
           / torch.sum(mask_inference_input)


def calculate_metric(pred_output_seqs, true_output_seqs):
    """
    Calculates metric based on Levenshtein distance. Lower and closer to 0
    metric values means better results.
    :param pred_output_seqs: batch predictions (mapping was applied), list
    of lists.
    :param true_output_seqs: sequence of np.array's with true output
    sequences from a dataframe.
    :return: mean Levenshtein distance proportional to sequences length
    """
    true_output_seqs = [''.join(sequence.tolist())
                        for sequence in true_output_seqs]
    pred_output_seqs = [''.join(sequence) for sequence in pred_output_seqs]
    levenshtein_values = np.array([
        levenshtein_distance(true_seq, pred_seq) /
            max(len(true_seq), len(pred_seq))
        for true_seq, pred_seq in zip(true_output_seqs, pred_output_seqs)
    ])
    return levenshtein_values.mean()


def display_metrics(metrics_history):
    """
    Display train loss and validation metric stored to the current step.
    :param metrics_history: dict[str: list] (metric -> values stored for every
    epoch to the current step) with fields "train_loss" and "val_levenshtein".
    """
    display.clear_output(wait=True)

    plt.figure(figsize=(12.8, 4.8))
    plt.subplot(1, 2, 1)
    line_train, = plt.plot(metrics_history['train_loss'], label='train loss')
    plt.title("Training loss")
    plt.xlabel("#epoch")
    plt.ylabel("loss")
    plt.legend(handles=[line_train])

    plt.subplot(1, 2, 2)
    line_val, = plt.plot(metrics_history['val_levenshtein'],
                         label='val levenshtein')
    plt.title("Val Levenshtein")
    plt.xlabel("#epoch")
    plt.ylabel("levenshtein")
    plt.legend(handles=[line_val])

    plt.show()


def train(model, train_data, val_data, optimizer, seq_max_len, epochs_count=30,
          batch_size=50):
    """
    Main train function, update model.
    :param model: pytorch model to be trained.
    :param train_data: dataframe with train data.
    :param val_data: dataframe with val data.
    :param optimizer: optimizer to use for training.
    :param seq_max_len: maximal sequence length.
    :param epochs_count: epochs count to train model.
    :param batch_size: batch size to use for training.
    :return: None
    """
    metrics_to_track = ["train_loss", "val_levenshtein"]

    metrics_history = {metric: [] for metric in metrics_to_track}

    for epoch in range(epochs_count):

        # Accumulates metrics for batches for ONE current epoch
        epoch_metrics = {metric: [] for metric in metrics_to_track}

        # Train part
        model.train()
        for pack in batch_generator(train_data, batch_size):
            one_hot_inputs, input_masks, one_hot_outputs = pack
            prediction = model.forward(one_hot_inputs, input_masks,
                                       one_hot_outputs=one_hot_outputs)
            loss = calculate_loss(prediction, input_masks, one_hot_outputs)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_metrics['train_loss'].append(loss.data.numpy())
        metrics_history['train_loss'].append(
            np.mean(epoch_metrics['train_loss']))

        # Evaluation part
        model.eval()
        for pack in batch_generator(val_data, batch_size, shuffle=False,
                                    return_source=True):
            one_hot_inputs, input_masks, _, input_seq, output_seq = pack
            prediction = model.forward(one_hot_inputs, input_masks,
                                       seq_max_len=seq_max_len)
            pred_output_seqs = model.apply_mapping(prediction)
            levenshtein_value = calculate_metric(pred_output_seqs,
                                                 output_seq)
            epoch_metrics['val_levenshtein'].append(levenshtein_value)
        metrics_history['val_levenshtein'].append(
            np.mean(epoch_metrics['val_levenshtein']))

        display_metrics(metrics_history)


def compare_sequences(model, data, seq_max_len, batch_size=4):
    """
    Print true and predicted sequences next to each other for first `batch_size`
    rows from the `data` dataframe.
    :param model: torch model to predict sequences.
    :param data: dataframe with data.
    :param seq_max_len: maximal sequence length.
    :param batch_size: number of samples to compare.
    :return: None
    """
    model.eval()

    for pack in batch_generator(data, batch_size, shuffle=False):
        one_hot_inputs, input_masks, one_hot_outputs = pack
        prediction = model.forward(one_hot_inputs, input_masks,
                                   seq_max_len=seq_max_len)
        break

    print("predicted sequences:\n", model.apply_mapping(prediction))
    print("true sequences:\n",
          [[sequence.tolist()] for sequence in data[:batch_size]["output"]])


def visualize_attention(model, data, seq_max_len, bos, eos, batch_size=2):
    """
    Attention visualization for all encoder and decoder states.
    :param model: torch model to predict sequences, contains field
    `attn_weights` with attention weights for all decoder states, all samples in
    the batch; list [decoder_states, Tensor(batch_size, encoder_states)]
    :param data: dataframe with data.
    :param seq_max_len: maximal sequence length.
    :param bos: bos symbol.
    :param eos: eos symbol.
    :param batch_size: number of samples to visualize attention.
    """
    model.eval()

    for pack in batch_generator(data, batch_size, shuffle=False):
        one_hot_inputs, input_masks, one_hot_outputs = pack
        predictions = model.forward(one_hot_inputs, input_masks,
                                    seq_max_len=seq_max_len)
        break

    batch_weights = model.attn_weights  # extract attention weights
    batch_input = data[:batch_size]["input"].values
    batch_output = data[:batch_size]["output"].values
    predictions = model.apply_mapping(predictions)

    for i, pack in enumerate(zip(input_masks, batch_input, batch_output)):
        mask_input, input_seq, output_seq = pack

        # weights for current sample
        weights = [weights_per_state[i, mask_input]
                   for weights_per_state in batch_weights]
        weights = np.stack(weights)
        weights = weights[:len(predictions[i]) + 2, :]

        fig, ax = plt.subplots()
        im = ax.imshow(weights, cmap="Greens")
        fig.colorbar(im, ax=ax)

        ax.set_xticks(range(weights.shape[1]))
        ax.set_yticks(range(len(predictions[i]) + 2))
        ax.set_xticklabels([bos] + list(input_seq) + [eos])
        ax.set_yticklabels([bos] + predictions[i] + [eos])
        ax.set_xlabel("Input")
        ax.set_ylabel("Prediction")

        plt.tight_layout()
        plt.show()
