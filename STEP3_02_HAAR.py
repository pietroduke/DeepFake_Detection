import cv2
import os
import glob
import time
start_time = time.time()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
path = glob.glob("Wider_samples/*.jpg")
saved = 0
for file in path:
    saved +=1
    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        try:
            face = img[y:y+h, x:x+h]
            p = os.path.sep.join(['Wider_faces/haar', "{}.jpg".format(saved)])
            cv2.imwrite(p, face)
            print("Đã lưu vào {}".format(p))
        except Exception as e:
            continue
end_time = time.time()
elapsed_time = end_time - start_time
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")