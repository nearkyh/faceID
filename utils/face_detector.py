import numpy as np
import cv2
import dlib
from ast import literal_eval


class FaceDetector:

    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor_path = 'utils/shape_predictor_68_face_landmarks.dat'
        self.predictor = dlib.shape_predictor(self.predictor_path)

    def face_detection(self, frame):
        detectors = self.detector(frame, 1)
        for i, d in enumerate(detectors):

            return {'x1':d.left(),
                    'y1':d.top(),
                    'x2':d.right(),
                    'y2':d.bottom()}

    def face_landmark_detection(self, frame):
        detectors = self.detector(frame, 1)
        for i, d in enumerate(detectors):
            shape = self.predictor(frame, d)
            xList = []
            yList = []
            for shapeNum in range(68):
                shapePoint = literal_eval(str(shape.part(shapeNum)))
                xList.append(shapePoint[0])
                yList.append(shapePoint[1])

            return {'x1':min(xList),
                    'y1':min(yList),
                    'x2':max(xList),
                    'y2':max(yList)}

    def visualization(self, frame):
        detectors = self.detector(frame, 1)

        # print("Number of faces detected: {}".format(len(detectors)))
        for i, d in enumerate(detectors):
            # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            #     i, d.left(), d.top(), d.right(), d.bottom()))
            cv2.putText(frame, 'faceID:{0}'.format(str('None')), (d.left(), d.top() - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 1)

            shape = self.predictor(frame, d)
            xList = []
            yList = []
            for shapeNum in range(68):
                shapePoint = literal_eval(str(shape.part(shapeNum)))
                cv2.circle(frame, shapePoint, 1, (255, 0, 0), -1)
                xList.append(shapePoint[0])
                yList.append(shapePoint[1])
            cv2.rectangle(frame, (min(xList), min(yList)), (max(xList), max(yList)), (255, 0, 0), 1)
