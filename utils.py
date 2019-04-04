# This file contains code modified licensed under the MIT License:
# Copyright (c) 2017 Guillaume Chevalier # For more information, visit:
# https://github.com/guillaume-chevalier/seq2seq-signal-prediction
# https://github.com/guillaume-chevalier/seq2seq-signal-prediction/blob/master/LICENSE

"""Contains functions to generate artificial data for predictions as well as
a function to plot predictions."""

import numpy as np
import Prediction
from random import randint
from matplotlib import pyplot as plt


def random_sine(series,effDataLen,batch_size, steps_per_epoch,
                input_sequence_length, target_sequence_length,
                min_frequency=0.1, max_frequency=10,
                min_amplitude=0.1, max_amplitude=1,
                min_offset=-0.5, max_offset=0.5,
                num_signals=3, seed=43):
    """Produce a batch of signals.

    The signals are the sum of randomly generated sine waves.

    Arguments
    ---------
    batch_size: Number of signals to produce.
    steps_per_epoch: Number of batches of size batch_size produced by the
        generator.
    input_sequence_length: Length of the input signals to produce.
    target_sequence_length: Length of the target signals to produce.
    min_frequency: Minimum frequency of the base signals that are summed.
    max_frequency: Maximum frequency of the base signals that are summed.
    min_amplitude: Minimum amplitude of the base signals that are summed.
    max_amplitude: Maximum amplitude of the base signals that are summed.
    min_offset: Minimum offset of the base signals that are summed.
    max_offset: Maximum offset of the base signals that are summed.
    num_signals: Number of signals that are summed together.
    seed: The seed used for generating random numbers

    Returns
    -------
    signals: 2D array of shape (batch_size, sequence_length)
    """

    startPoint = [45.494585811480725,-73.58088970184326]
    turningPoint = [45.494585811480725,-73.58088970184326]
    directDest = [45.55992553550895,-73.55403223653934]
    turnDest = [45.48593863930097,-73.5522973391931]


    batch_size = len(series)

    num_points = input_sequence_length + target_sequence_length
    x = np.arange(num_points) * 2 * np.pi / 30

    while True:
        # Reset seed to obtain same sequences from epoch to epoch
        np.random.seed(seed)

        for _ in range(steps_per_epoch):

            randIndexList = []
            for ele in range(len(effDataLen)):
                randIndexList.append(randint(1, int(effDataLen[ele])))
            print(randIndexList)

            srcArray = list()
            tgtArray = list()
            for x in range(len(series)):
                srcArray.append(series[x][:randIndexList[x], :])
                tgtArray.append(series[x][randIndexList[x] - 1:, :])

            encoder_input,__ = Prediction.align(srcArray)
            decoder_output,_ = Prediction.align(tgtArray)
            encoder_input =  np.asarray(encoder_input,dtype=series[0].dtype)
            decoder_output = np.asarray(decoder_output,dtype=series[0].dtype)

            # The output of the generator must be ([encoder_input, decoder_input], [decoder_output])
            decoder_input = np.zeros((decoder_output.shape[0], decoder_output.shape[1], decoder_output.shape[2]))


            yield ([encoder_input, decoder_input], decoder_output)


def plot_prediction(x, y_true, y_pred):
    """Plots the predictions.

    Arguments
    ---------
    x: Input sequence of shape (input_sequence_length,
        dimension_of_signal)
    y_true: True output sequence of shape (input_sequence_length,
        dimension_of_signal)
    y_pred: Predicted output sequence (input_sequence_length,
        dimension_of_signal)
    """

   # plt.figure(figsize=(12, 3))


    plt.plot(x[:, 2], x[:, 1], "o--b",
             label="existing")
    # plt.plot(range(len(past),
    # len(true) + len(past)), true, "x--b", label=label2)
    plt.plot(y_pred[:, 2], y_pred[:, 1], "o--y",
             label="pred")
    plt.legend(loc='best')
    plt.title("Predictions v.s. existing")

    while True:
        try:
            plt.show()
        except UnicodeDecodeError:
            continue
        break
