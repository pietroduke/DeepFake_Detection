# DeepFake_Detection
## **PHÁT HIỆN KHUÔN MẶT DEEPFAKE SỬ DỤNG MẠNG HỌC SÂU**
### Dataset
1) Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics. Source: https://github.com/yuezunli/celeb-deepfakeforensics
2) FaceForensics++: Learning to Detect Manipulated Facial Images. Source: https://github.com/ondyari/FaceForensics

### Thư viện
- Các thư viện cần cài đặt có thể tìm thấy trong `requirements.txt`
- Chạy `pip install -r requirements.txt` để cài đặt thư viện

### STEP 0: Phân chia video cho huấn luyện và kiểm thử của tệp FaceForensics++

*[Lưu ý]* Do tệp **FaceForensics++** mặc định không phân chia tệp test sẵn như **CELEB-DF-V2**, do đó cần phân chia video cho tệp Train : Test trước theo tỉ lệ: Train : Val : Test = 0.9 : 0.0 : 0.1 theo file `STEP2_split_dataset.py`, sau đó mới thực hiện các bước tiếp theo

### STEP 1: Chuyển đổi Video thành các Khung hình
Ở bước này, chỉ trích xuất số lượng khung hình nhất định. Cụ thể:

- Đối với tệp dữ liệu **CELEB-DF-V2**:
  - Tệp huấn luyện: 
    - Tệp REAL: Sau 6 Frames lấy 1 Frame
    - Tệp FAKE: Sau 40 Frames lấy 1 Frame
  - Tệp kiểm tra
    - Tệp REAL: Sau 4 Frames lấy 1 Frame
    - Tệp FAKE: Sau 8 Frames lấy 1 Frame
 
- Đối với tệp dữ liệu **FaceForensics++**:
  - Tệp huấn luyện: 
    - Tệp REAL: Sau 10 Frames lấy 1 Frame
    - Tệp FAKE: Sau 40 Frames lấy 1 Frame
  - Tệp kiểm tra
    - Tệp REAL: Sau 2 Frames lấy 1 Frame
    - Tệp FAKE: Sau 8 Frames lấy 1 Frame
 
### STEP 2: Phân chia dữ liệu cho tệp Huấn luyện (Train) và Đánh giá (Validate)
Phân chia theo tỉ lệ Train : Val = 0.8 : 0.2, có xáo trộn

### STEP 3: Sử dụng mô hình hệ thống nhận diện khuôn mặt để cắt khuôn mặt ra từ các khung hình
Ở đây sử dụng ba hệ thống nhận diện khuôn mặt chính để đánh giá hiệu suất cắt khuôn mặt của mỗi mô hình, thực hiện lần lượt các bước sau
1) Đầu tiên chạy các mã sau để lấy giá trị annotation của các phương pháp
  - Chạy `STEP3_01_DNN_face_extraction_from_frames.py` để sử dụng máy dò khuôn mặt DNN của OpenCV
  - Chạy `STEP3_02_HAAR.py` để sử dụng Haar Cascade
  - Chạy `STEP3_03_MTCNN.py` để sử dụng hệ thống dò mặt MTCNN
  Lưu kết quả đạt được, tiến hành chạy `wider_face_converter.py` và lưu các file `.txt` sinh ra ở `mAP_Wider/input/detection-results`
2) Chạy mã `wider_face_converter.py` để chuyển file `txt` annotation của bộ dữ liệu **Wider Face** về định dạng phù hợp với file đánh giá mAP của `mAP_Wider`, lưu các file `.txt` sinh ra ở `mAP_Wider/input/ground-truth`
3) Chạy mã `mAP_Wider/main.py` để đánh giá hiệu suất của ba phương pháp
4) Lựa chọn phương pháp tối ưu nhất (máy dò mặt của DNN trong OpenCV), sau đó chạy mã `STEP3_01_DNN_face_extraction_from_frames.py` một lần nữa với đường dẫn thư mục gốc là đường dẫn tới tập dữ liệu **FaceForensics++** và **CELEB-DF-V2**

### STEP 4: Huấn luyện mô hình và kiểm thử đối với tệp kiểm thử (Test)
Chạy mã `STEP4_train_test_CNN.ipynb` trên Google Colab để thực hiện quá trình huấn luyện mô hình.
- Trong quá trình huấn luyện có sử dụng:
  - Các mô hình pre-trained (DenseNet121, Xception, EfficientNetB0, EfficientNetB1, EfficientNetB2)
  - Data Generator (làm phong phú ảnh băng Data Augmentation)
  - Hàm loss: Categorical Classification
  - Số vòng huấn luyện: 20, Learning_rate = 0.0001, Batch_size = 32
- Trong quá trình kiểm thử có sử dụng:
  - Confusion matrix
  - Đồ thị ROC_curve
  - Script tính các thông số: AUC, EER, threshold_EER, TPR 0.5%, TPR 1%

### STEP 5: Đánh giá trên Video test của 2 tệp dataset, trên WEBCAM và video deepfake tự tạo
Kết quả đánh giá được thể hiện như hình dưới đây:
1) Video tập kiểm thử của CELEB-DF-V2

  - ![image](https://user-images.githubusercontent.com/75497356/154406385-ce7c7e14-0b0c-4490-891d-692188332569.png)

2) Video tập kiểm thử của FaceForensics++
  - ![image](https://user-images.githubusercontent.com/75497356/154406513-c3d48653-1c53-4414-a08f-3103f597890f.png)
3) Trên WEBCAM

  - ![image](https://user-images.githubusercontent.com/75497356/154406173-c4811bf8-b1fc-47fd-b024-f70d27d145e5.png)
4) Video deepfake tự tạo bằng DeepFakeLab

  - ![image](https://user-images.githubusercontent.com/75497356/154406593-afe02e27-4c3d-46ca-b247-687bdf7c472a.png)
 

