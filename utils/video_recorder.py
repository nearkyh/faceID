import numpy as np
import cv2


class VideoRecorder:

    def __init__(self):
        pass

    def set_record(self, fileName='test', width=640, height=480, frameRate=30.0):
        recording_video = "rec_{}.avi".format(fileName)
        fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')

        return cv2.VideoWriter(recording_video, fcc, frameRate, (width, height))

    def get_record(self, frame, set_record):
        set_record.write(frame)
