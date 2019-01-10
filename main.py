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


def predict_text_view(predict):
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    white = (255, 255, 255)
    black = (0, 0, 0)
    yellow = (0, 255, 255)
    magenta = (255, 0, 255)
    label_list, age_list, gender_list = dpp.get_labels(label_file_path='./face_recognition/data/labels.csv')
    x1, y1 =  (d.right() + 5), (d.top() - 2)
    x2, y2 = (d.right() + 80 + (len(label_list[predict]) * 10)) , (d.top() + 45)
    cv2.rectangle(image_np, (x1, y1), (x2, y2), black, -1)
    cv2.putText(image_np, 'faceID: {0}'.format(label_list[predict]), (d.right() + 10, d.top() + 15), cv2.FONT_HERSHEY_PLAIN, 1, yellow)
    cv2.putText(image_np, '{0}% Match'.format(predict_acc), (d.right() + 10, d.top() + 35), cv2.FONT_HERSHEY_PLAIN, 1, yellow)

def predict_box_view(frame):
    for shapePoint in range(len(shapePointQueue)):
        cv2.circle(frame, shapePointQueue[shapePoint], 1, (255, 0, 0), -1)
    cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 1)
    cv2.rectangle(frame, (min(xList), min(yList)), (max(xList), max(yList)), (255, 0, 0), 1)

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
            for i, d in enumerate(detectors):
                # ==================
                #   Face Detection
                # ==================
                # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                #     i, d.left(), d.top(), d.right(), d.bottom()))
                cropFaceDetection = image_np[d.top():d.bottom(), d.left():d.right()]
                # cv2.imshow('cropFaceDetection', cropFaceDetection)

                # ===================
                #   Shape Detection
                # ===================
                shape = shapeDetection(image_np, d)
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
                min_score_thresh = 30
                if (predict_acc >= min_score_thresh) and (predict_acc2 >= min_score_thresh):
                    if predict_acc > predict_acc2:
                        predict_text_view(predict=predict)
                        predict_box_view(frame=image_np)
                    elif predict_acc < predict_acc2:
                        predict_text_view(predict=predict2)
                        predict_box_view(frame=image_np)
                    elif predict_acc == predict_acc2:
                        predict_text_view(predict=predict)
                        predict_box_view(frame=image_np)
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
