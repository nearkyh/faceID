import numpy as np
import cv2
import math

from keras.models import load_model
from keras.utils.training_utils import multi_gpu_model

from utils.frame_rate import FrameRate

from face_recognition.data_preprocessing import DataPreProcessing


class Predictor:
    def __init__(self, model_path='./save_models/faceNet.h5'):
        self.model = load_model(model_path)
        self.dpp = DataPreProcessing()

    def run(self, frame):
        predict_image_np = self.dpp.input_frame(frame)
        predict = self.model.predict_classes(predict_image_np)
        predict = predict[0]
        prob = self.model.predict_proba(predict_image_np)
        prob = prob[0]
        predict_acc= prob[predict]
        predict_acc = float("{0:.2f}".format(predict_acc*100))

        return predict, predict_acc

    def score(self,
              img_data_path = '../face_recognition/data/img_data.npy',
              label_data_path = '../face_recognition/data/label_data.npy'):
        (x_train, x_test), (y_train, y_test) = self.dpp.get_data(img_data_path = img_data_path,
                                                                 label_data_path = label_data_path)
        score = self.model.evaluate(x_test, y_test, verbose=0)
        loss, accuracy = score[0], score[1]

        return loss, accuracy



if __name__ == '__main__':

    frameRate = FrameRate()
    predictor = Predictor()
    # dpp = DataPreProcessing()
    # label_list, age_list, gender_list = dpp.get_labels(label_file_path='./datasets/labels.csv')

    # test_img = dpp.input_image(img_path='./datasets/train/YongHan/0_0000.jpg')
    # model = load_model('./save_models/faceNet.h5')
    # predict = model.predict_classes(test_img)
    # print(predict)

    input_cam = 2
    cap = cv2.VideoCapture(input_cam)
    if cap.isOpened() == False:
        print('Can\'t open the CAM(%d)' % (input_cam))
        exit()

    while True:
        ret, image_np = cap.read()

        predict, predict_acc = predictor.run(image_np)

        frameRate.putText(image_np)
        cv2.putText(image_np, 'faceID: {0}, ACC: {1}%'.format(str(predict), predict_acc), (10, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
        cv2.imshow('faceID', image_np)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

    cap.release()
    cv2.destroyAllWindows()
