import load
import Prediction
import sys
import numpy as np
import Classification
import utils

test_folder = sys.argv[2]
suffix_name = Prediction.suffix_name
test_fileList = []

Prediction.getFileList(test_folder, test_fileList)

testSeries = list(map(load.loadFile, test_fileList))
testSeries, testEffDataLen = Prediction.align(testSeries)

test_input_data = np.asarray(testSeries, dtype=testSeries[0].dtype)




for seq_index in range(len(test_fileList)):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = test_input_data[seq_index: seq_index + 1]
    decoded_sentence = Prediction.predict(input_seq, Prediction.encoder_predict_model, Prediction.decoder_predict_model, Prediction.num_steps_to_predict)
    print('-')
    input_seq[:,:,0] *=86400
    input_seq[:,:,1] *=180
    input_seq[:,:,2] *=360

    print(input_seq)
    print('pre:', decoded_sentence)
    class_res = Classification.estimator.predict(decoded_sentence)
    print('classification result:', Classification.encoder.inverse_transform(class_res))
    utils.plot_prediction(input_seq[0, :, :], None, decoded_sentence[0, :, :])
