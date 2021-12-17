# DeepFake_Detection
## **PHÁT HIỆN KHUÔN MẶT DEEPFAKE DỰA TRÊN MẠNG HỌC SÂU**
### Dataset
1) Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics. Source: https://github.com/yuezunli/celeb-deepfakeforensics
2) FaceForensics++: Learning to Detect Manipulated Facial Images. Source: https://github.com/ondyari/FaceForensics

### Thư viện
Các thư viện cần cài đặt có thể tìm thấy trong `requirements.txt`
### STEP 0: Phân chia video cho huấn luyện và kiểm thử của tệp FaceForensics++

*[Lưu ý]* Do tệp **FaceForensics++** mặc định không phân chia tệp test sẵn như **CELEB-DF-V2**, do đó cần phân chia video cho tệp Train : Test trước theo tỉ lệ: Train : Val : Test = 0.9 : 0.0 : 0.1, sau đó mới thực hiện các bước tiếp theo

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

### STEP 3: Sử dụng mô hình cv2.dnn để cắt khuôn mặt ra từ các khung hình

### STEP 4: Huấn luyện mô hình và kiểm thử đối với tệp kiểm thử (Test)

### STEP 5: Đánh giá trên Video test của 2 tệp dataset và trên WEBCAM
