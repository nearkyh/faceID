import numpy as np
import cv2

from face_recognition.data_preprocessing import DataPreProcessing


class Visualization:

    def __init__(self, frame):
        self.frame = frame

        self.red = (0, 0, 255)
        self.green = (0, 255, 0)
        self.blue = (255, 0, 0)
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.yellow = (0, 255, 255)
        self.magenta = (255, 0, 255)

        self.thickness = 2
        self.line_range = 30
        self.text_font = cv2.FONT_HERSHEY_PLAIN
        self.text_size = 1

        self.dpp = DataPreProcessing()

    def prediction_box(self, predict, predict_acc, detect_point):
        dp = detect_point
        label_list, age_list, gender_list = self.dpp.get_labels(label_file_path='./face_recognition/data/labels.csv')
        textIndent = 10
        firstTextInterval = 20
        secondTextInterval = firstTextInterval + 20
        thirdTextInterval = secondTextInterval + 20
        fourthTextInterval = thirdTextInterval + 20
        boxLeft, boxTop = (dp.right() + 15, dp.top() - 2)
        boxRight, boxBottom = (dp.right() + 105 + (len(label_list[predict]) * 10), dp.top() + 30 + (fourthTextInterval - firstTextInterval))

        cv2.rectangle(self.frame, (boxLeft, boxTop), (boxRight, boxBottom), self.black, -1)
        cv2.putText(self.frame, '{0}% Match'.format(predict_acc), (boxLeft + textIndent, boxTop + firstTextInterval), self.text_font, self.text_size, self.yellow)
        cv2.putText(self.frame, 'faceID: {0}'.format(label_list[predict]), (boxLeft + textIndent, boxTop + secondTextInterval), self.text_font, self.text_size, self.yellow)
        cv2.putText(self.frame, 'Age: {0}'.format(age_list[predict]), (boxLeft + textIndent, boxTop + thirdTextInterval), self.text_font, self.text_size, self.yellow)
        cv2.putText(self.frame, 'Gender: {0}'.format(gender_list[predict]), (boxLeft + textIndent, boxTop + fourthTextInterval), self.text_font, self.text_size, self.yellow)

    def face_detection(self, detect_point):
        dp = detect_point
        # cv2.rectangle(frame, (dp.left(), dp.top()), (dp.right(), dp.bottom()), (0, 255, 0), 1)
        # point 1
        point_x1, point_y1 = dp.left(), dp.top()
        cv2.line(self.frame, (point_x1, point_y1), (point_x1, point_y1 + self.line_range), self.green, self.thickness)
        cv2.line(self.frame, (point_x1, point_y1), (point_x1 + self.line_range, point_y1), self.green, self.thickness)
        # point 2
        point_x2, point_y2 = dp.right(), dp.top()
        cv2.line(self.frame, (point_x2, point_y2), (point_x2, point_y2 + self.line_range), self.green, self.thickness)
        cv2.line(self.frame, (point_x2, point_y2), (point_x2 - self.line_range, point_y2), self.green, self.thickness)
        # point 3
        point_x3, point_y3 = dp.left(), dp.bottom()
        cv2.line(self.frame, (point_x3, point_y3), (point_x3, point_y3 - self.line_range), self.green, self.thickness)
        cv2.line(self.frame, (point_x3, point_y3), (point_x3 + self.line_range, point_y3), self.green, self.thickness)
        # point 4
        point_x4, point_y4 = dp.right(), dp.bottom()
        cv2.line(self.frame, (point_x4, point_y4), (point_x4, point_y4 - self.line_range), self.green, self.thickness)
        cv2.line(self.frame, (point_x4, point_y4), (point_x4 - self.line_range, point_y4), self.green, self.thickness)

    def shape_detection(self, shape_point, detect_point):
        shapePointQueue = shape_point
        for shapePoint in range(len(shapePointQueue)):
            cv2.circle(self.frame, shapePointQueue[shapePoint], 1, self.blue, -1)
        # cv2.rectangle(frame, (min(xList), min(yList)), (max(xList), max(yList)), line_color, 1)
        # point 1
        point_x1, point_y1 = detect_point['left'], detect_point['top']
        cv2.line(self.frame, (point_x1, point_y1), (point_x1, point_y1 + self.line_range), self.blue, self.thickness)
        cv2.line(self.frame, (point_x1, point_y1), (point_x1 + self.line_range, point_y1), self.blue, self.thickness)
        # point 2
        point_x2, point_y2 = detect_point['right'], detect_point['top']
        cv2.line(self.frame, (point_x2, point_y2), (point_x2, point_y2 + self.line_range), self.blue, self.thickness)
        cv2.line(self.frame, (point_x2, point_y2), (point_x2 - self.line_range, point_y2), self.blue, self.thickness)
        # point 3
        point_x3, point_y3 = detect_point['left'], detect_point['bottom']
        cv2.line(self.frame, (point_x3, point_y3), (point_x3, point_y3 - self.line_range), self.blue, self.thickness)
        cv2.line(self.frame, (point_x3, point_y3), (point_x3 + self.line_range, point_y3), self.blue, self.thickness)
        # point 4
        point_x4, point_y4 = detect_point['right'], detect_point['bottom']
        cv2.line(self.frame, (point_x4, point_y4), (point_x4, point_y4 - self.line_range), self.blue, self.thickness)
        cv2.line(self.frame, (point_x4, point_y4), (point_x4 - self.line_range, point_y4), self.blue, self.thickness)
