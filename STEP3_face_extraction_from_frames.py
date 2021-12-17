import numpy as np
import cv2
import os
import glob

# Khởi chạy mô hình crop face
print("Chạy crop faces...")
net = cv2.dnn.readNetFromCaffe('face_detector_dnn/deploy.prototxt',
                               'face_detector_dnn/res10_300x300_ssd_iter_140000.caffemodel')
# Khởi tạo đường dẫn folder dataset_frames
path = glob.glob("dataset_frames/CelebV2_Frames/test/real/*.jpg")
saved = 0
for file in path:
    saved +=1
    img = cv2.imread(file)
    #Nếu không phát hiện khuôn mặt trong frame thì bỏ qua
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
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            try:
                face = img[startY:endY, startX:endX]

                # Lưu ảnh khuôn mặt vào folder dataset
                p = os.path.sep.join(['dataset_faces/DatasetCELEB_test/real', "{}.jpg".format(saved)])
                cv2.imwrite(p, face)
                print("Đã lưu vào {}".format(p))
            except Exception as e:
                continue
