import cv2
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from threading import Thread
from PIL import Image, ImageTk

class ScrewDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Screw Detection")
        self.root.geometry("800x600")  # Fixed window size
        self.root.resizable(False, False)  # Disable resizing

        # Style the UI
        style = ttk.Style()
        style.configure("TButton", font=("Arial", 12), padding=10)
        style.configure("TLabel", font=("Arial", 14))

        # Initialize variables
        self.cap = None
        self.running = False
        self.model = YOLO("fp.pt")  # Load your YOLO model
        self.confidence_threshold = 0.75

        # Create UI layout
        self.video_frame = ttk.Label(root)
        self.video_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.control_frame = ttk.Frame(root)
        self.control_frame.pack(side=tk.BOTTOM, pady=10)

        self.start_button = ttk.Button(self.control_frame, text="Start Camera", command=self.start_camera)
        self.start_button.grid(row=0, column=0, padx=10)

        self.video_button = ttk.Button(self.control_frame, text="Play Video", command=self.play_video)
        self.video_button.grid(row=0, column=1, padx=10)

        self.stop_button = ttk.Button(self.control_frame, text="Stop", command=self.stop_detection, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=2, padx=10)

        self.quit_button = ttk.Button(self.control_frame, text="Quit", command=self.quit_app)
        self.quit_button.grid(row=0, column=3, padx=10)

    def start_camera(self):
        if not self.running:
            self.running = True
            self.cap = cv2.VideoCapture(0)  # Open webcam
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open webcam.")
                self.running = False
                return

            self.start_button.state(["disabled"])
            self.video_button.state(["disabled"])
            self.stop_button.state(["!disabled"])
            # Start a thread for detection
            self.thread = Thread(target=self.detect, daemon=True)
            self.thread.start()

    def play_video(self):
        if not self.running:
            video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mkv")])
            if not video_path:
                return  # User canceled the file dialog

            self.running = True
            self.cap = cv2.VideoCapture(video_path)  # Open video file
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open video.")
                self.running = False
                return

            self.start_button.state(["disabled"])
            self.video_button.state(["disabled"])
            self.stop_button.state(["!disabled"])
            # Start a thread for detection
            self.thread = Thread(target=self.detect, daemon=True)
            self.thread.start()

    def detect(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                messagebox.showinfo("Info", "Video finished or unable to read frame.")
                self.stop_detection()
                break

            # Perform YOLO detection
            results = self.model(frame, conf=self.confidence_threshold)
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    cls = int(box.cls[0])
                    label = self.model.names[cls]
                    conf_text = f"{conf:.2f}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf_text}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Convert the frame to RGB format for Tkinter display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)

            # Update the video feed
            self.video_frame.imgtk = imgtk
            self.video_frame.configure(image=imgtk)

    def stop_detection(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.start_button.state(["!disabled"])
        self.video_button.state(["!disabled"])
        self.stop_button.state(["disabled"])
        cv2.destroyAllWindows()
        self.video_frame.config(image="")  # Clear video feed

    def quit_app(self):
        self.stop_detection()
        self.root.destroy()

# Create the Tkinter app and run it
root = tk.Tk()
app = ScrewDetectionApp(root)
root.mainloop()
