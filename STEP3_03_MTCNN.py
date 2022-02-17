import cv2
import glob
import mtcnn
import os
import time
start_time = time.time()
path = glob.glob("path/to/origin/folder/*.jpg")
saved = 0
for file in path:
    saved += 1
    img = cv2.imread(file)
    try:
        (h, w) = img.shape[:2]
    except Exception as e:
        continue
    img = cv2.imread(file)
    detector = mtcnn.MTCNN()
    faces = detector.detect_faces(img)
    for face in faces:
        x, y, w, h = face['box']
        try:
            f = img[y:y+h, x:x+h]
            p = os.path.sep.join(['path/to/mtcnn_face/folder', "{}.jpg".format(saved)])
            cv2.imwrite(p, f)
            print("Đã lưu vào {}".format(p))
        except Exception as e:
            continue
end_time = time.time()
elapsed_time = end_time - start_time
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
