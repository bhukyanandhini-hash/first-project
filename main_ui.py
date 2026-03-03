import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
import sys
import threading
import time
import glob
import pandas as pd

# --- CONFIGURATION ---
INPUT_FOLDER = r"C:\Users\bhukya.nandhini\Documents\supertool\test_images"

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    from detector import your_detection_function
except ImportError:
    import detector
    your_detection_function = detector.your_detection_function

class QCApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Industrial QC - Metal Surface Inspector")
        self.window.geometry("1400x950")
        self.window.configure(bg="#121212")
        
        # Set Icon
        try:
            self.window.iconbitmap(resource_path(r"C:\Users\bhukya.nandhini\Documents\supertool\icon.ico"))
        except:
            pass
        tk.Label(window, text="AUTOMATIC QC MONITORING", fg="#00d4ff", bg="#121212", 
                 font=("Helvetica", 24, "bold")).pack(pady=10)
        
        self.status_lbl = tk.Label(window, text="Status: Ready", fg="#ffff00", bg="#121212", font=("Helvetica", 12))
        self.status_lbl.pack()

        # CONTROL PANEL
        self.btn_frame = tk.Frame(window, bg="#121212")
        self.btn_frame.pack(pady=10)

        self.is_monitoring = False
        self.start_btn = tk.Button(self.btn_frame, text="START MONITORING", bg="#28a745", fg="white", 
                                   font=("Helvetica", 12, "bold"), width=20, command=self.toggle_monitoring)
        self.start_btn.grid(row=0, column=0, padx=10)

        self.export_btn = tk.Button(self.btn_frame, text="EXPORT TO CSV", bg="#007bff", fg="white", 
                                    font=("Helvetica", 12, "bold"), width=20, command=self.export_to_csv)
        self.export_btn.grid(row=0, column=1, padx=10)

        # IMAGE DISPLAY
        self.pane = tk.Frame(window, bg="#121212")
        self.pane.pack(pady=5, fill=tk.BOTH, expand=True)

        self.canvas_ref = tk.Label(self.pane, bg="#1c1c1c", width=600, height=400)
        self.canvas_ref.grid(row=0, column=0, padx=20)
        
        self.canvas_det = tk.Label(self.pane, bg="#1c1c1c", width=600, height=400)
        self.canvas_det.grid(row=0, column=1, padx=20)

        # RESULT TABLE
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background="#1e1e1e", foreground="white", fieldbackground="#1e1e1e", rowheight=25)
        style.map("Treeview", background=[('selected', '#00d4ff')])

        self.tree_frame = tk.Frame(window, bg="#121212")
        self.tree_frame.pack(pady=10, fill=tk.X, padx=40)

        columns = ("id", "type", "length", "breadth", "file")
        self.tree = ttk.Treeview(self.tree_frame, columns=columns, show="headings", height=8)
        self.tree.heading("id", text="Defect ID")
        self.tree.heading("type", text="Defect Type")
        self.tree.heading("length", text="Length (mm)")
        self.tree.heading("breadth", text="Breadth (mm)")
        self.tree.heading("file", text="Source File")
        
        for col in columns:
            self.tree.column(col, width=150, anchor="center")
        self.tree.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.processed_files = set()
        self.all_defects = [] # Store data for CSV export

    def toggle_monitoring(self):
        if not self.is_monitoring:
            self.is_monitoring = True
            self.start_btn.config(text="STOP MONITORING", bg="#dc3545")
            self.status_lbl.config(text="Status: Monitoring Folder...", fg="#00ff00")
            threading.Thread(target=self.folder_watcher, daemon=True).start()
        else:
            self.is_monitoring = False
            self.start_btn.config(text="START MONITORING", bg="#28a745")
            self.status_lbl.config(text="Status: Stopped", fg="#ffff00")

    def folder_watcher(self):
        while self.is_monitoring:
            files = glob.glob(os.path.join(INPUT_FOLDER, "*.[jJ][pP]*[gG]"))
            for file_path in files:
                if not self.is_monitoring: break
                if file_path not in self.processed_files:
                    filename = os.path.basename(file_path)
                    self.window.after(0, lambda p=filename: self.status_lbl.config(text=f"Processing: {p}"))
                    
                    ref, det, data = your_detection_function(file_path)
                    
                    if ref is not None:
                        self.window.after(0, lambda r=ref, d=det: self.update_display(r, d))
                        self.window.after(0, lambda d_log=data, f=filename: self.update_table(d_log, f))
                    
                    self.processed_files.add(file_path)
                    time.sleep(2)
            time.sleep(1)

    def update_display(self, ref_img, det_img):
        self.show_img(ref_img, self.canvas_ref)
        self.show_img(det_img, self.canvas_det)

    def update_table(self, data, filename):
        """Clears view for current image and adds to master list."""
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        for entry in data:
            try:
                # Format: "#1 Fine Line: 4.86mm x 0.86mm"
                parts = entry.split(" ", 1) 
                d_id = parts[0]
                details = parts[1].split(": ") 
                d_type = details[0]
                dims = details[1].split(" x ") 
                length = dims[0].replace("mm", "")
                breadth = dims[1].replace("mm", "")
                
                # Add to UI
                self.tree.insert("", tk.END, values=(d_id, d_type, length, breadth, filename))
                
                # Add to Master List for CSV
                self.all_defects.append({
                    "Defect ID": d_id,
                    "Type": d_type,
                    "Length (mm)": length,
                    "Breadth (mm)": breadth,
                    "File": filename
                })
            except:
                pass

    def export_to_csv(self):
        if not self.all_defects:
            messagebox.showwarning("No Data", "No defects found to export.")
            return
        
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            df = pd.DataFrame(self.all_defects)
            df.to_csv(file_path, index=False)
            messagebox.showinfo("Success", f"Data exported successfully to {os.path.basename(file_path)}")

    def show_img(self, img_array, label):
        rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img.thumbnail((600, 400), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        label.config(image=img_tk)
        label.image = img_tk 

if __name__ == "__main__":
    root = tk.Tk()
    app = QCApp(root)
    root.mainloop()