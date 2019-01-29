from __future__ import print_function

import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

from data_preprocessing import DataPreProcessing
from models.faceNet import FaceNet


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

def history_visualization(model_info):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # summarize history for accuracy
    axs[0].plot(range(1, len(model_info.history['acc']) + 1), model_info.history['acc'])
    axs[0].plot(range(1, len(model_info.history['val_acc']) + 1), model_info.history['val_acc'])
    axs[0].set_title('model_info Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_info.history['acc']) + 1), len(model_info.history['acc']) / 10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1, len(model_info.history['loss']) + 1), model_info.history['loss'])
    axs[1].plot(range(1, len(model_info.history['val_loss']) + 1), model_info.history['val_loss'])
    axs[1].set_title('model_info Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_info.history['loss']) + 1), len(model_info.history['loss']) / 10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()



if __name__ == '__main__':

    dpp = DataPreProcessing()
    (x_train, x_test), (y_train, y_test) = dpp.load_data(img_data_path='./data/img_data.npy',
                                                         label_data_path='./data/label_data.npy')
    (x_train, x_test), (y_train, y_test) = dpp.pre_processing(x_train, x_test, y_train, y_test)
    label_list, age_list, gender_list = dpp.get_labels()
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    faceNet = FaceNet(input_shape=x_train.shape[1:],
                      num_classes=len(label_list),
                      gpu=True)
    model = faceNet.build()
    model_info = model.fit(x_train, y_train,
                           batch_size=batch_size,
                           epochs=epochs,
                           verbose=1,
                           validation_data=(x_test, y_test),
                           callbacks=[faceNet.tensorboard()])
    model.summary()
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    history_visualization(model_info=model_info)

    try:
        if not(os.path.isdir('./save_models')):
            os.makedirs(os.path.join('./save_models'))
    except Exception as e:
        print("Failed to create directory!!!")
    model.save('./save_models/faceNet.h5')
