
# File này dùng để kiểm tra camera có lấy được hình ảnh hay không

import cv2
import numpy as np

# Khởi tạo kết nối với camera
cap = cv2.VideoCapture(0)

# Kiểm tra kết nối
if not cap.isOpened():
    print("Không thể kết nối với camera")
    exit()

while True:

    ret, frame = cap.read()
    frame = cv2.resize(frame, (int(800), int(600)))

    # Nếu không đọc được khung hình, thoát khỏi vòng lặp
    if not ret:
        print("Không thể đọc khung hình từ camera")
        break

    cv2.imshow('Camera Feed', frame)

    # Nhấn phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()