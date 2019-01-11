import numpy as np
import cv2
import dlib
import sys
import os
from ast import literal_eval

from utils.frame_rate import FrameRate

from face_recognition.predictor import Predictor
from face_recognition.data_preprocessing import DataPreProcessing
# from face_recognition.data_creator.face_capture import FaceCapture


class Visualization:

    def __init__(self, frame):
        self.frame = frame

        self.thickness = 2
        self.line_range = 30

        self.red = (0, 0, 255)
        self.green = (0, 255, 0)
        self.blue = (255, 0, 0)
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.yellow = (0, 255, 255)
        self.magenta = (255, 0, 255)

    def prediction(self, predict):
        label_list, age_list, gender_list = dpp.get_labels(label_file_path='./face_recognition/data/labels.csv')
        x1, y1 = (detect.right() + 5), (detect.top() - 2)
        x2, y2 = (detect.right() + 80 + (len(label_list[predict]) * 10)), (detect.top() + 45)
        cv2.rectangle(image_np, (x1, y1), (x2, y2), self.black, -1)
        cv2.putText(image_np, 'faceID: {0}'.format(label_list[predict]), (detect.right() + 10, detect.top() + 15), cv2.FONT_HERSHEY_PLAIN, 1, self.yellow)
        cv2.putText(image_np, '{0}% Match'.format(predict_acc), (detect.right() + 10, detect.top() + 35), cv2.FONT_HERSHEY_PLAIN, 1, self.yellow)

    def face_detection(self):
        # cv2.rectangle(frame, (detect.left(), detect.top()), (detect.right(), detect.bottom()), (0, 255, 0), 1)
        # point 1
        point_x1, point_y1 = detect.left(), detect.top()
        cv2.line(self.frame, (point_x1, point_y1), (point_x1, point_y1 + self.line_range), self.green, self.thickness)
        cv2.line(self.frame, (point_x1, point_y1), (point_x1 + self.line_range, point_y1), self.green, self.thickness)
        # point 2
        point_x2, point_y2 = detect.right(), detect.top()
        cv2.line(self.frame, (point_x2, point_y2), (point_x2, point_y2 + self.line_range), self.green, self.thickness)
        cv2.line(self.frame, (point_x2, point_y2), (point_x2 - self.line_range, point_y2), self.green, self.thickness)
        # point 3
        point_x3, point_y3 = detect.left(), detect.bottom()
        cv2.line(self.frame, (point_x3, point_y3), (point_x3, point_y3 - self.line_range), self.green, self.thickness)
        cv2.line(self.frame, (point_x3, point_y3), (point_x3 + self.line_range, point_y3), self.green, self.thickness)
        # point 4
        point_x4, point_y4 = detect.right(), detect.bottom()
        cv2.line(self.frame, (point_x4, point_y4), (point_x4, point_y4 - self.line_range), self.green, self.thickness)
        cv2.line(self.frame, (point_x4, point_y4), (point_x4 - self.line_range, point_y4), self.green, self.thickness)

    def shape_detection(self):
        for shapePoint in range(len(shapePointQueue)):
            cv2.circle(self.frame, shapePointQueue[shapePoint], 1, self.blue, -1)
        # cv2.rectangle(frame, (min(xList), min(yList)), (max(xList), max(yList)), line_color, 1)
        # point 1
        point_x1, point_y1 = min(xList), min(yList)
        cv2.line(self.frame, (point_x1, point_y1), (point_x1, point_y1 + self.line_range), self.blue, self.thickness)
        cv2.line(self.frame, (point_x1, point_y1), (point_x1 + self.line_range, point_y1), self.blue, self.thickness)
        # point 2
        point_x2, point_y2 = max(xList), min(yList)
        cv2.line(self.frame, (point_x2, point_y2), (point_x2, point_y2 + self.line_range), self.blue, self.thickness)
        cv2.line(self.frame, (point_x2, point_y2), (point_x2 - self.line_range, point_y2), self.blue, self.thickness)
        # point 3
        point_x3, point_y3 = min(xList), max(yList)
        cv2.line(self.frame, (point_x3, point_y3), (point_x3, point_y3 - self.line_range), self.blue, self.thickness)
        cv2.line(self.frame, (point_x3, point_y3), (point_x3 + self.line_range, point_y3), self.blue, self.thickness)
        # point 4
        point_x4, point_y4 = max(xList), max(yList)
        cv2.line(self.frame, (point_x4, point_y4), (point_x4, point_y4 - self.line_range), self.blue, self.thickness)
        cv2.line(self.frame, (point_x4, point_y4), (point_x4 - self.line_range, point_y4), self.blue, self.thickness)


# Define Face Detection
faceDetection = dlib.get_frontal_face_detector()
shape_predictor_path = 'face_detection/shape_predictor_68_face_landmarks.dat'
shapeDetection = dlib.shape_predictor(shape_predictor_path)

# Define Face Recognition
dpp = DataPreProcessing()
faceRecognition = Predictor(model_path='./face_recognition/save_models/faceNet.h5')

# Define Utils
frameRate = FrameRate()
# faceCapture = FaceCapture()



if __name__ == '__main__':

    faceID = 0
    savePath = 'faceDB'

    input_cam = 2
    cap = cv2.VideoCapture(input_cam)
    if cap.isOpened() == False:
        print('Can\'t open the CAM(%d)' % (input_cam))
        exit()

    while True:
        ret, image_np = cap.read()
        original_frame = image_np

        # image_np = cv2.flip(image_np, 1)  # 좌우 반전

        # 데이터셋 구축을 위한 얼굴 이미지 캡쳐 기능
        '''
        try:
            faceCapture.run(frame=image_np,
                            faceID=faceID,
                            savePath=savePath)
        except Exception as e:
            pass
        '''

        try:
            detectors = faceDetection(image_np, 1)
            # print("Number of faces detected: {}".format(len(detectors)))
            for num, detect in enumerate(detectors):
                # ==================
                #   Face Detection
                # ==================
                # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                #     num, detect.left(), detect.top(), detect.right(), detect.bottom()))
                cropFaceDetection = image_np[detect.top():detect.bottom(), detect.left():detect.right()]
                # cv2.imshow('cropFaceDetection', cropFaceDetection)

                # ===================
                #   Shape Detection
                # ===================
                shape = shapeDetection(image_np, detect)
                xList = []
                yList = []
                shapePointQueue = []
                for shapeNum in range(68):
                    shapePoint = literal_eval(str(shape.part(shapeNum)))
                    shapePointQueue.append(shapePoint)
                    xList.append(shapePoint[0])
                    yList.append(shapePoint[1])
                cropFaceLandmark =  image_np[min(yList): max(yList), min(xList):max(xList)]
                # cv2.imshow('cropFaceLandmark', cropFaceLandmark)

                # ====================
                #   Face Recognition
                # ====================
                predict, predict_acc = faceRecognition.run(cropFaceDetection)
                predict2, predict_acc2 = faceRecognition.run(cropFaceLandmark)

                # =================
                #   Visualization
                # =================
                visualization = Visualization(frame=image_np)
                min_score_thresh = 30
                if (predict_acc >= min_score_thresh) and (predict_acc2 >= min_score_thresh):
                    if predict_acc > predict_acc2:
                        visualization.prediction(predict=predict)
                        visualization.face_detection()
                        visualization.shape_detection()
                    elif predict_acc < predict_acc2:
                        visualization.prediction(predict=predict2)
                        visualization.face_detection()
                        visualization.shape_detection()
                    elif predict_acc == predict_acc2:
                        visualization.prediction(predict=predict)
                        visualization.face_detection()
                        visualization.shape_detection()
                else:
                    pass


        except Exception as e:
            pass

        frameRate.putText(frame=image_np)
        cv2.imshow('faceID', image_np)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

    cap.release()
    cv2.destroyAllWindows()
