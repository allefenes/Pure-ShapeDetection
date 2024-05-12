import cv2
import numpy as np

def detect_shapes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gauss filtresi uyguladım
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptif eşikleme uyguladım
    thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Kernel oluşturup önce aşındırdım sonra genişlettim
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded = cv2.erode(thresholded, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_shapes = []

    for contour in contours:
        area = cv2.contourArea(contour)
        #Bu area değeri 100 den büyük olanları içeriyordu 3000 yaptım ki arkaplandaki vs nesneleride anlamlandırmasın.
        if area > 3000:     
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
            num_vertices = len(approx)
            
            shape = ""
            if num_vertices == 3:
                shape = "Üçgen"
            elif num_vertices == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                if 0.95 <= aspect_ratio <= 1.05:
                    shape = "Kare"
                else:
                    shape = "Dikdörtgen"
            elif num_vertices == 10:
                shape = "Yildiz"
            elif num_vertices > 4:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                if radius > 20:
                    shape = "Daire"
                else:
                    shape = "Oval"
            
            cv2.drawContours(image, [contour], 0, (0, 255, 0), 2)
            
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(image, shape, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
                center_x = cx - (image.shape[1] // 2)
                center_y = cy - (image.shape[0] // 2)
                detected_shapes.append((shape, (center_x, center_y)))
    
    return image, detected_shapes

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    detected_frame, shapes = detect_shapes(frame)
    cv2.imshow('Detected Shapes', detected_frame)
    for shape, (center_x, center_y) in shapes:
        print(f"{shape} merkezinin x konumu: {center_x}, y konumu: {center_y}")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()