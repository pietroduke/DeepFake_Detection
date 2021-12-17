from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import pickle
import cv2
from tensorflow_addons.layers import SpatialPyramidPooling2D
from efficientnet.tfkeras import EfficientNetB0

thresh = 0.5

# Khởi chạy mô hình nhận diện khuôn mặt
net = cv2.dnn.readNetFromCaffe('face_detector_dnn/deploy.prototxt',
                               'face_detector_dnn/res10_300x300_ssd_iter_140000.caffemodel')
# Load mô hình đã đào tạo trên Google Colab
model = load_model('G:/report/celeb_result/spp_densenet121/SPP_DenseNet121_CELEB.h5',
				   custom_objects={'SpatialPyramidPooling2D': SpatialPyramidPooling2D})
# Load nhãn đã được mã hoá (Label Encoder)
le = pickle.loads(open('G:/project/mix_xception/le.pickle', "rb").read())

#  Đọc video từ thư mục test và save lại video sau khi test
cap = cv2.VideoCapture('dataset_videos/celeb-v2/test/fake/id0_id3_0007.mp4')
fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
out = cv2.VideoWriter("output_video.avi", fourcc, 30, (1366, 768))

while (cap.isOpened()):
	# Đọc lần lượt từng frame
	_,frame = cap.read()
	frame = cv2.resize(frame, (1366, 768))
	# Chuyển thành blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
	# Phát hiện khuôn mặt trong frame
	net.setInput(blob)
	detections = net.forward()

	# Loop qua các khuôn mặt
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		# Nếu ngưỡng lấy khuôn mặt threshold thì khởi tạo tham số để tiến hành đóng khung face:
		if confidence > thresh:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			startX = max(0, startX)
			startY = max(0, startY)
			endX = min(w, endX)
			endY = min(h, endY)

			# Lấy vùng khuôn mặt
			face = frame[startY:endY, startX:endX]
			try:
				face = cv2.resize(face, (224,224))
			except Exception as e:
				continue
			face = face.astype("float") / 255.0
			face = img_to_array(face)
			face = np.expand_dims(face, axis=0)

			# Model sẽ dự đoán khuôn mặt là real/deepfake
			preds = model.predict(face)[0]
			j = np.argmax(preds)
			label = le.classes_[j]

			# Vẽ box hình chữ nhật đóng khung khuôn mặt
			label = "{}: {:.4f}".format(label, preds[j])
			if (j==0):
				# Vẽ màu đỏ nếu khuôn mặt là FAKE
				cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
				cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
			else:
				# Vẽ màu xanh nếu khuôn mặt là REAL
				cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
				cv2.rectangle(frame, (startX, startY), (endX, endY), (0,  255,0), 2)

	out.write(frame)
	#Hiện cửa sổ console
	cv2.namedWindow("Detect Face", cv2.WINDOW_NORMAL)
	cv2.imshow("Detect Face", frame)
	if cv2.waitKey(1) == 27:
		break

#Clean chương trình
cap.release()
out.release()
cv2.destroyAllWindows()