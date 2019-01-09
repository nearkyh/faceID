import numpy as np
import cv2
import os

from keras.models import load_model
from utils.frame_rate import FrameRate
from face_recognition.datasets.face_data import FaceData


class Predictor:
    def __init__(self, model_path='./save_models/faceNet.h5'):
        self.model = load_model(model_path)
        self.faceData = FaceData()

    def run(self, frame):
        predict_image_np = self.faceData.input_frame(frame)
        predict = self.model.predict_classes(predict_image_np)

        return predict



if __name__ == '__main__':

    frameRate = FrameRate()
    predictor = Predictor()
    # faceData = FaceData()
    # labels = faceData.get_labels(label_file_path='./datasets/labels.csv')

    # test_img = faceData.input_image(img_path='./datasets/train/YongHan/0_0000.jpg')
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

        predict = predictor.run(image_np)

        frameRate.putText(image_np)
        cv2.putText(image_np, 'faceID:{0}'.format(str(predict[0])), (10, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
        cv2.imshow('faceID', image_np)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

    cap.release()
    cv2.destroyAllWindows()
