import numpy as np
import cv2
import dlib
import sys
import os
from ast import literal_eval

from utils.frame_rate import FrameRate
from utils.visualization import Visualization

from face_recognition.predictor import Predictor
from face_recognition.data_preprocessing import DataPreProcessing
# from face_recognition.data_creator.face_capture import FaceCapture


# Define Face Detection
faceDetection = dlib.get_frontal_face_detector()
shape_predictor_path = 'face_detection/shape_predictor_68_face_landmarks.dat'
shapeDetection = dlib.shape_predictor(shape_predictor_path)

# Define Face Recognition
faceRecognition = Predictor(model_path='./face_recognition/save_models/faceNet.h5')
dpp = DataPreProcessing()

# Define Utils
frameRate = FrameRate()
# faceCapture = FaceCapture()
# visualization = Visualization()



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
                shape_detect = {'left':min(xList), 'top':min(yList), 'right':max(xList), 'bottom':max(yList)}
                cropFaceLandmark = image_np[shape_detect['top']: shape_detect['bottom'], shape_detect['left']:shape_detect['right']]
                # cropFaceLandmark = image_np[min(yList): max(yList), min(xList):max(xList)]
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
                        visualization.prediction_box(predict=predict,
                                                     predict_acc=predict_acc,
                                                     detect_point=detect)
                        visualization.face_detection(detect_point=detect)
                        visualization.shape_detection(shape_point=shapePointQueue,
                                                      detect_point=shape_detect)
                    elif predict_acc < predict_acc2:
                        visualization.prediction_box(predict=predict2,
                                                     predict_acc=predict_acc2,
                                                     detect_point=detect)
                        visualization.face_detection(detect_point=detect)
                        visualization.shape_detection(shape_point=shapePointQueue,
                                                      detect_point=shape_detect)
                    elif predict_acc == predict_acc2:
                        visualization.prediction_box(predict=predict,
                                                     predict_acc=predict_acc,
                                                     detect_point=detect)
                        visualization.face_detection(detect_point=detect)
                        visualization.shape_detection(shape_point=shapePointQueue,
                                                      detect_point=shape_detect)
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
