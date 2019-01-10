import numpy as np
import cv2
import dlib
from ast import literal_eval
import os
import time


class FaceDataCreator:

    def __init__(self):
        self.saveCount = 1
        self.saveNum = 500
        self.faceDetection = dlib.get_frontal_face_detector()
        self.shapeDetection_path = '../face_detection/shape_predictor_68_face_landmarks.dat'
        self.shapeDetection = dlib.shape_predictor(self.shapeDetection_path)

    def run(self, frame, faceID, savePath):
        detectors = self.faceDetection(frame, 1)
        for i, d in enumerate(detectors):
            # Create directory
            saveDir = '{0}/{1}/'.format(savePath, faceID)
            if not os.path.isdir(saveDir):
                os.mkdir(saveDir)
                print("Create directory({})".format(saveDir))

            # Face Detection
            cropFaceDetection = frame[d.top():d.bottom(), d.left():d.right()]

            # DATA_1
            imgName = "{0}_{1}.jpg".format(str(faceID), "{0:04d}".format(self.saveCount))
            cv2.imwrite(saveDir + imgName, cropFaceDetection)
            print("Save[DATA_1]", saveDir + imgName)
            self.saveCount += 1

            # DATA_2
            imgName = "{0}_{1}.jpg".format(str(faceID), "{0:04d}".format(self.saveCount))
            flipCropFaceDetection = cv2.flip(cropFaceDetection, 1)  # 좌우 반전
            cv2.imwrite(saveDir + imgName, flipCropFaceDetection)
            print("Save[DATA_2]", saveDir + imgName)
            self.saveCount += 1

            # Shape Detection
            shape = self.shapeDetection(frame, d)
            xList = []
            yList = []
            for shapeNum in range(68):
                shapePoint = literal_eval(str(shape.part(shapeNum)))
                xList.append(shapePoint[0])
                yList.append(shapePoint[1])
            cropFaceLandmark =  frame[min(yList): max(yList), min(xList):max(xList)]

            # DATA_3
            imgName = "{0}_{1}.jpg".format(str(faceID), "{0:04d}".format(self.saveCount))
            cv2.imwrite(saveDir + imgName, cropFaceLandmark)
            print("Save[DATA_3]", saveDir + imgName)
            self.saveCount += 1

            # DATA_4
            imgName = "{0}_{1}.jpg".format(str(faceID), "{0:04d}".format(self.saveCount))
            flipCropFaceLandmark = cv2.flip(cropFaceLandmark, 1)  # 좌우 반전
            cv2.imwrite(saveDir + imgName, flipCropFaceLandmark)
            print("Save[DATA_4]", saveDir + imgName)
            self.saveCount += 1

            # Visualization
            cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 1)
            cv2.rectangle(frame, (min(xList), min(yList)), (max(xList), max(yList)), (255, 0, 0), 1)

            # 500장 저장후 종료
            if len(os.walk(saveDir).__next__()[2]) >= self.saveNum:
                exit()



if __name__ == '__main__':

    faceDataCreator = FaceDataCreator()

    faceID = 'KangHyeon'
    savePath = '../face_recognition/datasets'

    input_cam = 2
    cap = cv2.VideoCapture(input_cam)
    if cap.isOpened() == False:
        print('Can\'t open the CAM(%d)' % (input_cam))
        exit()

    while True:
        ret, image_np = cap.read()

        # 데이터셋 구축을 위한 얼굴 이미지 캡쳐 기능
        try:
            faceDataCreator.run(frame=image_np,
                                faceID=faceID,
                                savePath=savePath)
        except Exception as e:
            print(e)
            pass

        cv2.imshow('Face Data Creator', image_np)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

    cap.release()
    cv2.destroyAllWindows()
