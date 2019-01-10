import numpy as np
import cv2
import os
import csv

from keras.utils import np_utils


class DataPreProcessing:

    def __init__(self):
        self.train_data_path = '../face_recognition/datasets/train/'
        self.test_data_path = '../face_recognition/datasets/test/'
        self.label_file_path = '../face_recognition/data/labels.csv'
        self.image_w = 96
        self.image_h = 96

    def get_labels(self, label_file_path='../face_recognition/data/labels.csv'):
        with open(label_file_path, 'r', encoding='utf-8') as f:
            data = csv.reader(f)
            labels = []
            for line in data:
                labels.append(line[1])

        return labels

    def pre_processing(self,
                       label_file_path='../face_recognition/data/labels.csv',
                       train_data_path='../face_recognition/datasets/train/',
                       test_data_path='../face_recognition/datasets/test/'):
        labels = self.get_labels()
        num_classes = len(labels)

        x_train = []  # train data
        y_train = []  # train labels
        x_test = []  # test data
        y_test = []  # test labels

        for index, label in enumerate(labels):
            train_image_dir = train_data_path + label + '/'
            test_image_dir = test_data_path + label + '/'

            for top, dir, f in os.walk(train_image_dir):
                for filename in f:
                    train_img = cv2.imread(train_image_dir + filename)
                    train_img = cv2.resize(train_img, None, fx=self.image_w / train_img.shape[1], fy=self.image_h / train_img.shape[0])
                    x_train.append(train_img)
                    y_train.append(index)

            for top, dir, f in os.walk(test_image_dir):
                for filename in f:
                    test_img = cv2.imread(test_image_dir + filename)
                    test_img = cv2.resize(test_img, None, fx=self.image_w / test_img.shape[1], fy=self.image_h / test_img.shape[0])
                    x_test.append(test_img)
                    y_test.append(index)

        x_train = np.array(x_train)
        x_test = np.array(x_test)

        y_train = np.array(y_train)
        y_train = y_train[np.newaxis]   # 차원수 증가 1 -> 2
        y_train = y_train.transpose()   # 행 -> 열 변환
        y_test = np.array(y_test)
        y_test = y_test[np.newaxis]     # 차원수 증가 1 -> 2
        y_test = y_test.transpose()     # 행 -> 열 변환

        # 형병환
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        # 정규화
        x_train /= 255
        x_test /= 255

        # 원-핫 벡터
        y_train = np_utils.to_categorical(y_train, num_classes)
        y_test = np_utils.to_categorical(y_test, num_classes)

        img_data_path = '../face_recognition/data/img_data.npy'
        label_data_path = '../face_recognition/data/label_data.npy'
        np.save(img_data_path, (x_train, x_test))
        np.save(label_data_path, (y_train, y_test))

        if os.path.exists(img_data_path):
            print("Successfully created the img_data.npy")
        if os.path.exists(label_data_path):
            print("Successfully created the label_data.npy")

        # return x_train, x_test, y_train, y_test

    def get_data(self,
                 img_data_path = '../face_recognition/data/img_data.npy',
                 label_data_path = '../face_recognition/data/label_data.npy'):
        x_train, x_test = np.load(img_data_path)
        y_train, y_test = np.load(label_data_path)

        return (x_train, x_test), (y_train, y_test)

    def input_image(self, img_path):
        dataList = []
        img = cv2.imread(img_path)
        img = cv2.resize(img, None, fx=self.image_w / img.shape[1], fy=self.image_h / img.shape[0])
        dataList.append(img)
        dataList = np.array(dataList)
        dataList = dataList.astype('float32')
        dataList /= 255

        return dataList

    def input_frame(self, frame):
        dataList = []
        img = cv2.resize(frame, None, fx=self.image_w / frame.shape[1], fy=self.image_h / frame.shape[0])
        dataList.append(img)
        dataList = np.array(dataList)
        dataList = dataList.astype('float32')
        dataList /= 255

        return dataList



if __name__ == '__main__':

    dpp = DataPreProcessing()
    dpp.pre_processing()
    (x_train, x_test), (y_train, y_test) = dpp.get_data()

    print(x_train.shape)
