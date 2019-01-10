from __future__ import print_function

import os
import argparse

from face_recognition.data_preprocessing import DataPreProcessing
from face_recognition.models.faceNet import FaceNet


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='face', type=str,
                    help="Input face data")
parser.add_argument('--batch_size', default=32, type=int,
                    help="Number of batch_size")
parser.add_argument('--epochs', default=10, type=int,
                    help="Number of epochs")
parser.add_argument('--gpu', default=False, type=bool,
                    help="Using GPU")
args = parser.parse_args()
batch_size = args.batch_size
epochs = args.epochs



if __name__ == '__main__':

    dpp = DataPreProcessing()
    (x_train, x_test), (y_train, y_test) = dpp.get_data(img_data_path='./data/img_data.npy',
                                                        label_data_path='./data/label_data.npy')
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    faceNet = FaceNet(input_shape=x_train.shape[1:],
                      num_classes=3,
                      gpu=True)
    model = faceNet.build()

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    try:
        if not(os.path.isdir('./save_models')):
            os.makedirs(os.path.join('./save_models'))
    except Exception as e:
        print("Failed to create directory!!!")
    model.save('./save_models/faceNet.h5')
