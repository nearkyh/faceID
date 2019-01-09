import numpy as np
import cv2
import dlib

from ast import literal_eval

from utils.frame_rate import FrameRate

from face_recognition.predictor import Predictor
from face_recognition.datasets.face_data import FaceData
# from face_recognition.data_creator.face_capture import FaceCapture


# Define Face Detection
faceDetection = dlib.get_frontal_face_detector()
shape_predictor_path = 'face_detection/shape_predictor_68_face_landmarks.dat'
shapeDetection = dlib.shape_predictor(shape_predictor_path)

# Define Face Recognition
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

        # ================================
        #   Face Detection & Recognition
        # ================================
        detectors = faceDetection(image_np, 1)
        # print("Number of faces detected: {}".format(len(detectors)))
        for i, d in enumerate(detectors):
            # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            #     i, d.left(), d.top(), d.right(), d.bottom()))
            cv2.rectangle(image_np, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 1)

            # ====================
            #   Face Recognition
            # ====================
            cropFaceDetection = image_np[d.top():d.bottom(), d.left():d.right()]
            predict = faceRecognition.run(cropFaceDetection)
            cv2.putText(image_np, 'faceID:{0}'.format(str(predict[0])), (10, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            cv2.putText(image_np, 'faceID:{0}'.format(str(predict[0])), (d.left(), d.top() - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            # cv2.putText(image_np, 'faceID:{0}'.format(str('None')), (d.left(), d.top() - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

            # ===================
            #   Shape Detection
            # ===================
            shape = shapeDetection(image_np, d)
            xList = []
            yList = []
            for shapeNum in range(68):
                shapePoint = literal_eval(str(shape.part(shapeNum)))
                cv2.circle(image_np, shapePoint, 1, (255, 0, 0), -1)
                xList.append(shapePoint[0])
                yList.append(shapePoint[1])
            cv2.rectangle(image_np, (min(xList), min(yList)), (max(xList), max(yList)), (255, 0, 0), 1)
            cropFaceLandmark =  image_np[min(yList): max(yList), min(xList):max(xList)]

        frameRate.putText(frame=image_np)
        cv2.imshow('faceID', image_np)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

    cap.release()
    cv2.destroyAllWindows()
