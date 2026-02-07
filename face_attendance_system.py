import os
import csv
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import spacy

DATASET_DIR = "dataset"
TRAINER_DIR = "trainer"
ATTENDANCE_DIR = "attendance"
MODEL_PATH = os.path.join(TRAINER_DIR, "face_model.h5")
LABELS_PATH = os.path.join(TRAINER_DIR, "labels.csv")

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(TRAINER_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

try:
    nlp = spacy.load("en_core_web_sm")
except:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def preprocess_image_numpy(img_array):
    img_normalized = np.array(img_array, dtype=np.float32) / 255.0
    mean, std = np.mean(img_normalized), np.std(img_normalized)
    img_standardized = (img_normalized - mean) / (std + 1e-7)
    return img_standardized


def save_labels_pandas(name_to_id):
    df = pd.DataFrame(list(name_to_id.items()), columns=['Name', 'ID'])
    df['Registration_Date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df.to_csv(LABELS_PATH, index=False)


def load_labels_pandas():
    if os.path.exists(LABELS_PATH):
        df = pd.read_csv(LABELS_PATH)
        return dict(zip(df['ID'], df['Name']))
    return {}


def mark_attendance_pandas(name):
    today = datetime.now().strftime("%Y-%m-%d")
    csv_path = os.path.join(ATTENDANCE_DIR, f"attendance_{today}.csv")
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if name in df['Name'].values:
            return False
    else:
        df = pd.DataFrame(columns=['Name', 'Time', 'Date'])
    
    new_record = pd.DataFrame({
        'Name': [name],
        'Time': [datetime.now().strftime("%H:%M:%S")],
        'Date': [today]
    })
    df = pd.concat([df, new_record], ignore_index=True)
    df.to_csv(csv_path, index=False)
    return True


def generate_stats_pandas():
    files = [f for f in os.listdir(ATTENDANCE_DIR) if f.startswith("attendance_") and f.endswith(".csv")]
    if not files:
        return None
    
    dfs = [pd.read_csv(os.path.join(ATTENDANCE_DIR, f)) for f in files]
    combined = pd.concat(dfs, ignore_index=True)
    stats = combined.groupby('Name').agg({'Date': 'count', 'Time': 'first'}).rename(
        columns={'Date': 'Total_Days', 'Time': 'First_Time'})
    return stats.reset_index()


def create_cnn_model(num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model_tensorflow():
    image_paths, labels, name_to_id = [], [], {}
    next_id = 0
    
    for file in os.listdir(DATASET_DIR):
        if not file.lower().endswith(".jpg"):
            continue
        name = file.rsplit(".", 1)[0].rsplit("_", 1)[0]
        if name not in name_to_id:
            name_to_id[name] = next_id
            next_id += 1
        image_paths.append(os.path.join(DATASET_DIR, file))
        labels.append(name_to_id[name])
    
    if not image_paths:
        return None, {}
    
    X_train = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (100, 100))
        img = preprocess_image_numpy(img)
        X_train.append(img)
    
    X_train = np.expand_dims(np.array(X_train), axis=-1)
    y_train = np.array(labels)
    
    model = create_cnn_model(len(name_to_id))
    early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=20, batch_size=8, callbacks=[early_stop], verbose=1)
    model.save(MODEL_PATH)
    
    id_to_name = {v: k for k, v in name_to_id.items()}
    save_labels_pandas(name_to_id)
    return model, id_to_name


def load_model_tensorflow():
    if os.path.exists(MODEL_PATH):
        return keras.models.load_model(MODEL_PATH), load_labels_pandas()
    return None, {}


def detect_and_process_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    return faces, gray


def preprocess_face_opencv(face_roi):
    face = cv2.resize(face_roi, (100, 100))
    face = cv2.equalizeHist(face)
    face = cv2.GaussianBlur(face, (5, 5), 0)
    return face


def generate_nlp_report(name):
    text = f"{name} marked present at {datetime.now().strftime('%H:%M:%S')} today."
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return text, entities


class FaceAttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Attendance System")
        self.root.geometry("900x700")
        self.root.configure(bg="#1a1a2e")
        
        self.cap = None
        self.running = False
        self.mode = "attendance"
        self.current_name = None
        self.capture_count = 0
        self.max_captures = 30
        self.model, self.labels = load_model_tensorflow()
        
        self.build_ui()
    
    def build_ui(self):
        title = tk.Label(self.root, text="FACE ATTENDANCE SYSTEM", font=("Arial", 22, "bold"),
                         bg="#16213e", fg="#00d4ff")
        title.pack(fill=tk.X, pady=10)
        
        btn_frame = tk.Frame(self.root, bg="#1a1a2e")
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="REGISTER", width=15, height=2, bg="#4a90e2", fg="white",
                  font=("Arial", 11, "bold"), command=self.enter_register).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="ATTENDANCE", width=15, height=2, bg="#27ae60", fg="white",
                  font=("Arial", 11, "bold"), command=self.enter_attendance).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="STATISTICS", width=15, height=2, bg="#9b59b6", fg="white",
                  font=("Arial", 11, "bold"), command=self.show_stats).pack(side=tk.LEFT, padx=5)
        
        self.video_label = tk.Label(self.root, bg="#2c3e50", text="CAMERA OFF",
                                    fg="white", font=("Arial", 16))
        self.video_label.pack(pady=20, expand=True)
        
        self.status_label = tk.Label(self.root, text="Ready", bg="#1a1a2e", fg="#00ff88",
                                     font=("Arial", 12))
        self.status_label.pack(pady=5)
        
        control = tk.Frame(self.root, bg="#1a1a2e")
        control.pack(pady=10)
        tk.Button(control, text="START", width=12, height=2, bg="#e74c3c", fg="white",
                  font=("Arial", 11, "bold"), command=self.start_camera).pack(side=tk.LEFT, padx=5)
        tk.Button(control, text="STOP", width=12, height=2, bg="#7f8c8d", fg="white",
                  font=("Arial", 11, "bold"), command=self.stop_camera).pack(side=tk.LEFT, padx=5)
    
    def enter_register(self):
        self.mode = "register"
        self.status_label.config(text="REGISTER MODE")
        self.ask_name()
    
    def enter_attendance(self):
        self.mode = "attendance"
        self.status_label.config(text="ATTENDANCE MODE")
        self.model, self.labels = load_model_tensorflow()
    
    def ask_name(self):
        popup = tk.Toplevel(self.root)
        popup.title("Register")
        popup.geometry("300x150")
        popup.configure(bg="#16213e")
        popup.grab_set()
        
        tk.Label(popup, text="Enter Name:", bg="#16213e", fg="white",
                 font=("Arial", 12)).pack(pady=15)
        entry = tk.Entry(popup, font=("Arial", 12), width=20)
        entry.pack(pady=5)
        entry.focus()
        
        def save():
            name = entry.get().strip()
            if name:
                self.current_name = name
                self.capture_count = 0
                popup.destroy()
                self.status_label.config(text=f"Registering: {name}")
        
        tk.Button(popup, text="OK", bg="#27ae60", fg="white", font=("Arial", 11),
                  width=10, command=save).pack(pady=10)
    
    def start_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Cannot access camera")
                self.cap = None
                return
        self.running = True
        self.update_frame()
    
    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.config(image="", text="CAMERA OFF")
    
    def update_frame(self):
        if not self.running or not self.cap:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.stop_camera()
            return
        
        frame = cv2.flip(frame, 1)
        faces, gray = detect_and_process_face(frame)
        
        if self.mode == "register" and self.current_name:
            self.handle_register(frame, gray, faces)
        elif self.mode == "attendance" and self.model:
            self.handle_attendance(frame, gray, faces)
        
        cv2.putText(frame, self.mode.upper(), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (640, 480))
        img = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self.video_label.imgtk = img
        self.video_label.configure(image=img, text="")
        
        self.root.after(15, self.update_frame)
    
    def handle_register(self, frame, gray, faces):
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face = preprocess_face_opencv(gray[y:y+h, x:x+w])
            
            if self.capture_count < self.max_captures:
                self.capture_count += 1
                cv2.imwrite(os.path.join(DATASET_DIR, f"{self.current_name}_{self.capture_count}.jpg"), face)
                cv2.putText(frame, f"{self.capture_count}/{self.max_captures}", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                self.stop_camera()
                self.status_label.config(text="Training model...")
                self.root.update()
                self.model, self.labels = train_model_tensorflow()
                self.status_label.config(text=f"{self.current_name} registered successfully")
                messagebox.showinfo("Success", f"{self.current_name} has been registered")
            break
    
    def handle_attendance(self, frame, gray, faces):
        for (x, y, w, h) in faces:
            face = preprocess_face_opencv(gray[y:y+h, x:x+w])
            face_input = np.expand_dims(np.expand_dims(preprocess_image_numpy(face), axis=0), axis=-1)
            
            predictions = self.model.predict(face_input, verbose=0)
            pred_id = np.argmax(predictions)
            confidence = np.max(predictions)
            
            if confidence > 0.7 and pred_id in self.labels:
                name = self.labels[pred_id]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if mark_attendance_pandas(name):
                    report, _ = generate_nlp_report(name)
                    self.status_label.config(text=f"{name} marked present")
                else:
                    self.status_label.config(text=f"{name} already marked today")
            else:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, "UNKNOWN", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            break
    
    def show_stats(self):
        stats = generate_stats_pandas()
        if stats is None or stats.empty:
            messagebox.showinfo("Statistics", "No attendance data available")
            return
        
        win = tk.Toplevel(self.root)
        win.title("Attendance Statistics")
        win.geometry("600x400")
        win.configure(bg="#16213e")
        
        tk.Label(win, text="ATTENDANCE STATISTICS", font=("Arial", 16, "bold"),
                 bg="#16213e", fg="#00d4ff").pack(pady=10)
        
        text = tk.Text(win, font=("Courier", 10), bg="#2c3e50", fg="white", height=15, width=70)
        text.pack(pady=10, padx=20)
        text.insert("1.0", stats.to_string(index=False))
        text.insert("end", f"\n\nTotal Students: {len(stats)}\nTotal Records: {stats['Total_Days'].sum()}")
        text.config(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceAttendanceApp(root)
    root.mainloop()
