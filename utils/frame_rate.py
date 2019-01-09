import numpy as np
import cv2
import time


class FrameRate:

    def __init__(self):
        self.prevTime = 0

    def putText(self, frame):
        curTime = time.time()
        sec = curTime - self.prevTime
        self.prevTime = curTime
        fps = 1 / (sec)
        frameRate = "FPS : %0.1f" % fps
        cv2.putText(frame, frameRate, (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
