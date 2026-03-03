import cv2
import numpy as np
import os

# --- CONFIGURATION ---
REF_PATH = r"C:\Users\bhukya.nandhini\Desktop\Image processing\ref\side tale part.jpg"
REFERENCE_WIDTH_MM = 300 # Used to calculate pixels per mm

def your_detection_function(defect_path):
    """
    Analyzes images for defects and returns aligned reference, 
    annotated test image, and a list of measurements.
    """
    ref_img = cv2.imread(REF_PATH)
    def_img = cv2.imread(defect_path)
    
    if ref_img is None or def_img is None: 
        return None, None, ["Error: Could not load images. Check paths."]

    h, w = def_img.shape[:2]
    ref_img = cv2.resize(ref_img, (w, h))
    ppm = w / REFERENCE_WIDTH_MM # Pixels per millimeter

    # 1. Alignment (ORB)
    gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    gray_def = cv2.cvtColor(def_img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(1000) 
    kp1, des1 = orb.detectAndCompute(gray_ref, None)
    kp2, des2 = orb.detectAndCompute(gray_def, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
    
    if len(matches) < 10: 
        return ref_img, def_img, ["Alignment Failed: Too few matches."]
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if matrix is None:
        return ref_img, def_img, ["Matrix calculation failed."]
        
    ref_aligned = cv2.warpPerspective(ref_img, matrix, (w, h))

    # 2. Difference Analysis (HSV Red-Masking)
    hsv = cv2.cvtColor(ref_aligned, cv2.COLOR_BGR2HSV)
    lower_red1, upper_red1 = np.array([0, 60, 50]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 60, 50]), np.array([180, 255, 255])
    red_mask = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), 
                              cv2.inRange(hsv, lower_red2, upper_red2))
    
    detection_mask = cv2.erode(red_mask, np.ones((30, 30), np.uint8), iterations=1)
    diff = cv2.absdiff(cv2.cvtColor(ref_aligned, cv2.COLOR_BGR2GRAY), gray_def)
    _, thresh = cv2.threshold(cv2.GaussianBlur(diff, (5, 5), 0), 35, 255, cv2.THRESH_BINARY)
    final_diff = cv2.bitwise_and(thresh, detection_mask)

    # 3. Classification & Measurement
    contours, _ = cv2.findContours(final_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_img = def_img.copy()
    defect_log = []

    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) < 5: continue 
        rect = cv2.minAreaRect(cnt)
        (x_c, y_c), (w_box, h_box), _ = rect
        length_mm = max(w_box, h_box) / ppm
        breadth_mm = min(w_box, h_box) / ppm
        
        if length_mm < 0.3: continue 

        # Classify by Aspect Ratio
        aspect_ratio = length_mm / (breadth_mm + 0.001)
        if aspect_ratio > 2.2:
            label = "Fine Line"
            box = cv2.boxPoints(rect).astype(int)
            cv2.drawContours(output_img, [box], 0, (0, 255, 255), 3)
        else:
            label = "Dot"
            center = (int(x_c), int(y_c))
            radius = int(max(w_box, h_box) / 1.5)
            cv2.circle(output_img, center, radius, (0, 255, 255), 3)
        
        # Format string for UI parsing
        defect_log.append(f"#{i+1} {label}: {length_mm:.2f}mm x {breadth_mm:.2f}mm")
        cv2.putText(output_img, f"#{i+1}", (int(x_c), int(y_c) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    return ref_aligned, output_img, defect_log