import time
import cv2
import glob
import numpy as np
import os

start_time = time.time()

net = cv2.dnn.readNetFromCaffe('DNN_face_detector/deploy.prototxt',
                               'DNN_face_detector/res10_300x300_ssd_iter_140000.caffemodel')
path = glob.glob("path/to/origin/folder/*.jpg")
saved = 0
for file in path:
    saved +=1
    img = cv2.imread(file)
    try:
        (h, w) = img.shape[:2]
    except Exception as e:
        continue
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    if len(detections) > 0:
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]
        if confidence > 0.55:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            try:
                face = img[startY:endY, startX:endX]
                p = os.path.sep.join(['path/to/dnn_face/folder', "{}.jpg".format(saved)])
                cv2.imwrite(p, face)
                print("Đã lưu vào {}".format(p))
            except Exception as e:
                continue
end_time = time.time()
elapsed_time = end_time - start_time
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
