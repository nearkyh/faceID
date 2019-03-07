# Face ID


## Requirements
- Ubuntu 16.04
- Python 3.5
- OpenCV
- Dlib
- Keras(Tensorflow backend)


## Getting Started
Creating virtualenv
```bash
$ cd faceID
$ virtualenv env --python=python3.5
$ source env/bin/activate
```

Install Dependencies
```bash
$ pip install -r requirements.txt
```

Run
```bash
$ python main.py
```


## Face Detection
- Using [Dlib](http://dlib.net/) library.
- In order to use the face shape landmarks detection, you have to unzip [the file](https://github.com/yonghankim/faceID/blob/master/face_detection/shape_predictor_68_face_landmarks.dat.bz2).
#### How to use
```python
import dlib, cv2

fileName = 'face.jpg'
img = cv2.imread(fileName)
predictorPath = 'face_detection/shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictorPath)
dets = detector(img, 1)

print("Number of faces detected: {}".format(len(dets)))
for i, d in enumerate(dets):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        i, d.left(), d.top(), d.right(), d.bottom()))
    shape = predictor(img, d)
    print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                              shape.part(1)))
```


## Face Recognition
- Implementation of CNN on Python3, Keras(Tensorflow backend)
