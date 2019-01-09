import numpy as np
from PIL import Image
import os


class PreProcessing:

    def __init__(self):
        self.dataPath = '../faceDB/'

        self.dirName = '../faceDB/0/'
        self.fileNames = os.listdir('../faceDB/0/')
        self.labelFile = '../dataset/labels.txt'

    def precess(self):
        for i in self.fileNames:
            filePath = os.path.join(self.dirName, i)
            img = Image.open(filePath)

            # 이미지 사이즈 고정 (128, 128)
            imgSize = (128, 128)
            resizeImg = img.resize(imgSize)
            resizeImg.save('{0}resize_{1}'.format(self.dirName, i))

            # 이미지 좌우 반전
            flipImg = resizeImg.transpose(Image.FLIP_LEFT_RIGHT)
            flipImg.save('{0}rotate_{1}'.format(self.dirName, i))

            # 원본 파일 삭제
            os.remove(filePath)

    def read_data(self):
        dirList = os.listdir(self.dataPath)
        for label in dirList:
            dataPath = os.path.join(self.dataPath, label)   # Data path full name
            print('label:',label)
            faceData = os.listdir(dataPath)
            for data in faceData:
                faceData = os.path.join(dataPath, data)  # Data path full name
                # print(faceData)
                img = Image.open(faceData).convert("L")
                arr = np.array(img)
                print(arr)

    def load_labels(self):
        with open(self.labelFile, 'r') as f:
            row = f.read()
            print(row)



if __name__ == '__main__':

    preProcessing = PreProcessing()
    print(preProcessing.read_data())
    #
    # a = os.listdir('../faceDB/')
    # print(a)
    # a = list(map(int, a))
    # print(a)
    # b = sorted(a)
    # print(b)
