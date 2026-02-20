import cv2
import numpy as np
import pandas as pd
import threading
import os
import glob
import time
import io
from flask import Flask, render_template_string, Response, send_file

app = Flask(__name__)

# --- CONFIGURATION ---
REF_PATH = r"C:\Users\bhukya.nandhini\Desktop\Image processing\ref\IMG-20251203-WA0000(1).jpg"
INPUT_FOLDER = r"C:\Users\bhukya.nandhini\Desktop\Image processing\test"
REFERENCE_WIDTH_MM = 300

# --- GLOBAL STATE ---
# Stores the latest image and defect list to keep the UI responsive
latest_results = {"img": None, "data": [], "count": 0, "status": "Initializing..."}

# --- 1. THE DETECTION ENGINE (Integrated Logic) ---
def detect_defects_logic(ref_path, defect_path):
    ref_img = cv2.imread(ref_path)
    def_img = cv2.imread(defect_path)
    if ref_img is None or def_img is None: 
        return None, []
    
    h, w = def_img.shape[:2]
    ref_img = cv2.resize(ref_img, (w, h))
    ppm = w / REFERENCE_WIDTH_MM 

    # Alignment (ORB)
    gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    gray_def = cv2.cvtColor(def_img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(1000) 
    kp1, des1 = orb.detectAndCompute(gray_ref, None)
    kp2, des2 = orb.detectAndCompute(gray_def, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
    
    if len(matches) < 10: return def_img, [] 
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    ref_aligned = cv2.warpPerspective(ref_img, matrix, (w, h))

    # Difference Analysis
    hsv = cv2.cvtColor(ref_aligned, cv2.COLOR_BGR2HSV)
    lower_red1, upper_red1 = np.array([0, 60, 50]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 60, 50]), np.array([180, 255, 255])
    red_mask = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), 
                              cv2.inRange(hsv, lower_red2, upper_red2))
    detection_mask = cv2.erode(red_mask, np.ones((30, 30), np.uint8), iterations=1)
    
    diff = cv2.absdiff(cv2.cvtColor(ref_aligned, cv2.COLOR_BGR2GRAY), gray_def)
    _, thresh = cv2.threshold(cv2.GaussianBlur(diff, (5, 5), 0), 35, 255, cv2.THRESH_BINARY)
    final_diff = cv2.bitwise_and(thresh, detection_mask)

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

        # CLASSIFICATION & MARKING (Circles for Dots, Boxes for Lines)
        aspect_ratio = length_mm / (breadth_mm + 0.001)
        if aspect_ratio > 2.2:
            label = "Fine Line"
            box = cv2.boxPoints(rect).astype(int)
            cv2.drawContours(output_img, [box], 0, (0, 255, 255), 3) # Box for Line
        else:
            label = "Dot"
            center = (int(x_c), int(y_c))
            radius = int(max(w_box, h_box) / 1.5)
            cv2.circle(output_img, center, radius, (0, 255, 255), 3) # Circle for Dot
        
        defect_log.append({
            "ID": i + 1,
            "Type": label,
            "Length (mm)": round(length_mm, 2),
            "Breadth (mm)": round(breadth_mm, 2)
        })
        cv2.putText(output_img, f"#{i+1}", (int(x_c), int(y_c) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    return output_img, defect_log

# --- 2. BACKGROUND WORKER ---
def worker():
    global latest_results
    while True:
        files = glob.glob(os.path.join(INPUT_FOLDER, "*.jp*g"))
        if not files:
            latest_results["status"] = "Waiting for images..."
            time.sleep(2)
            continue
            
        for f in files:
            latest_results["status"] = f"Processing {os.path.basename(f)}..."
            img, data = detect_defects_logic(REF_PATH, f)
            if img is not None:
                latest_results["img"] = img
                latest_results["data"] = data
                latest_results["count"] = len(data)
            time.sleep(1.5)

# --- 3. WEB INTERFACE ---
@app.route('/')
def index():
    html = """
    <html>
        <head>
            <title>Industrial QC Monitor</title>
            <meta http-equiv="refresh" content="2">
            <style>
                body { font-family: sans-serif; display: flex; margin: 0; background: #121212; color: #e0e0e0; }
                .sidebar { width: 350px; padding: 20px; background: #1e1e1e; height: 100vh; border-right: 2px solid #00d4ff; }
                .main { flex: 1; padding: 20px; text-align: center; }
                table { width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 14px; }
                th, td { border: 1px solid #333; padding: 10px; text-align: left; }
                th { background: #00d4ff; color: #000; }
                .img-container { border: 3px solid #333; border-radius: 10px; padding: 5px; background: #000; display: inline-block; max-width: 95%; }
                .btn { display: inline-block; margin-top: 20px; padding: 12px; background: #28a745; color: white; text-decoration: none; border-radius: 5px; font-weight: bold; width: 100%; text-align: center; }
            </style>
        </head>
        <body>
            <div class="sidebar">
                <h2 style="color:#00d4ff">QC DASHBOARD</h2>
                <p>Status: <b style="color:#00ff00">{{ status }}</b></p>
                <hr style="border:0.5px solid #333">
                <h3>Current Defects: <span style="color:#ff4b4b">{{ count }}</span></h3>
                <table>
                    <tr><th>ID</th><th>Type</th><th>L (mm)</th><th>B (mm)</th></tr>
                    {% for item in data %}
                    <tr><td>{{ item.ID }}</td><td>{{ item.Type }}</td><td>{{ item['Length (mm)'] }}</td><td>{{ item['Breadth (mm)'] }}</td></tr>
                    {% endfor %}
                </table>
                <a href="/export" class="btn">📥 DOWNLOAD CSV REPORT</a>
            </div>
            <div class="main">
                <h1>LIVE INSPECTION FEED</h1>
                <div class="img-container">
                    <img src="/feed" style="max-width: 100%; height: auto;">
                </div>
            </div>
        </body>
    </html>
    """
    return render_template_string(html, **latest_results)

@app.route('/feed')
def feed():
    if latest_results["img"] is not None:
        _, buffer = cv2.imencode('.jpg', latest_results["img"])
        return Response(buffer.tobytes(), mimetype='image/jpeg')
    return "No Feed"

@app.route('/export')
def export():
    if not latest_results["data"]: return "No data", 400
    df = pd.DataFrame(latest_results["data"])
    output = io.BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return send_file(output, mimetype="text/csv", as_attachment=True, download_name="Defect_Report.csv")

if __name__ == '__main__':
    threading.Thread(target=worker, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False)