import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import torch
import numpy as np
from PIL import Image, ImageTk
import threading
from torchvision import transforms
from models.csrnet import CSRNet
from ultralytics import YOLO
import scipy.ndimage
import pygame 

class CrowdCountingGUI:
    def strip_module_prefix(self, state_dict):
        if any(k.startswith('module.') for k in state_dict.keys()):
            return {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        return state_dict

    def __init__(self, root):
        self.root = root
        self.root.title("Crowd Counting Application")
        self.root.title("Crowd Counting Application")
        self.root.geometry("600x500") # Reduced size since video is separate
        self.root.resizable(True, True)
        self.video_window = None # Toplevel window for video
        
        # Variables
        self.model_choice = tk.StringVar(value="csrnet")
        self.view_choice = tk.StringVar(value="front") # Front or Top view
        self.input_choice = tk.StringVar(value="file")
        self.video_path = tk.StringVar(value="")
        self.is_processing = False
        self.cap = None
        self.current_frame = None
        self.threshold_value = tk.DoubleVar(value=0.022)  # Default CSRNet threshold
        self.count_threshold = tk.IntVar(value=100)  # Default count threshold for alerts
        self.alert_active = False  # Track if alert is currently showing
        self.flash_state = False # Track flash state for color toggling
        self.camera_index = tk.IntVar(value=0) # Camera index selection
        
        # ROI Variables
        self.roi_coords = None # (x1, y1, x2, y2)
        self.roi_active = False
        self.roi_select_active = False
        self.roi_start = None
        self.roi_current = None
        self.original_threshold = None # To store threshold before scaling
        self.roi_coords = None # (x1, y1, x2, y2)
        self.roi_active = False
        self.roi_select_active = False
        self.roi_start = None
        self.roi_current = None
        self.original_threshold = None # To store threshold before scaling
        self.image_item = None # Canvas image item ID
        
        # Display Scaling Variables
        self.display_scale = 1.0
        self.display_offset_x = 0
        self.display_offset_y = 0

        
        # Model paths
        self.csrnet_path_front = "best_csrnet.pth"
        self.csrnet_path_top = "best_csrnet_top.tar"
        self.yolo_path_front = "best.pt"
        self.yolo_path_top = "best_top.pt"

        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Models (loaded on demand)
        self.csrnet_model = None
        self.yolo_model = None
        
        # CSRNet transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Fixed dimensions
        self.fixed_width = 1280
        self.fixed_height = 720
        
        # Initialize Audio
        try:
            pygame.mixer.init()
            self.sound_enabled = True
        except Exception as e:
            print(f"Audio initialization failed: {e}")
            self.sound_enabled = False

        # Create GUI
        self.create_widgets()
        
    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Control Panel
        control_frame = ttk.LabelFrame(main_frame, text="Configuration", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Model Selection
        model_frame = ttk.Frame(control_frame)
        model_frame.pack(fill=tk.X, pady=5)
        ttk.Label(model_frame, text="Select Model:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(model_frame, text="CSRNet (Density Map)", variable=self.model_choice, value="csrnet").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(model_frame, text="YOLOv8 (Object Detection)", variable=self.model_choice, value="yolo").pack(side=tk.LEFT, padx=5)

        
        # View Selection
        view_frame = ttk.Frame(control_frame)
        view_frame.pack(fill=tk.X, pady=5)
        ttk.Label(view_frame, text="Select View:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(view_frame, text="Front View", variable=self.view_choice, value="front").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(view_frame, text="Top View", variable=self.view_choice, value="top").pack(side=tk.LEFT, padx=5)
        
        # Input Selection
        input_frame = ttk.Frame(control_frame)
        input_frame.pack(fill=tk.X, pady=5)
        ttk.Label(input_frame, text="Select Input:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(input_frame, text="Video File", variable=self.input_choice, value="file", command=self.on_input_change).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(input_frame, text="Camera", variable=self.input_choice, value="camera", command=self.on_input_change).pack(side=tk.LEFT, padx=5)
        
        # Camera Index Selection
        ttk.Label(input_frame, text="Index:").pack(side=tk.LEFT, padx=(10, 2))
        self.camera_spinbox = ttk.Spinbox(input_frame, from_=0, to=10, textvariable=self.camera_index, width=3)
        self.camera_spinbox.pack(side=tk.LEFT, padx=2)


        
        # File Selection
        file_frame = ttk.Frame(control_frame)
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(file_frame, text="Video File:").pack(side=tk.LEFT, padx=5)
        self.file_entry = ttk.Entry(file_frame, textvariable=self.video_path, width=50)
        self.file_entry.pack(side=tk.LEFT, padx=5)
        self.browse_btn = ttk.Button(file_frame, text="Browse...", command=self.browse_file)
        self.browse_btn.pack(side=tk.LEFT, padx=5)


        
        # CSRNet Threshold Slider
        self.threshold_frame = ttk.Frame(control_frame)
        self.threshold_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(self.threshold_frame, text="CSRNet Threshold:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        self.threshold_slider = ttk.Scale(self.threshold_frame, from_=0.001, to=0.100, variable=self.threshold_value, orient=tk.HORIZONTAL, length=200)
        self.threshold_slider.pack(side=tk.LEFT, padx=5)
        self.threshold_label = ttk.Label(self.threshold_frame, text=f"{self.threshold_value.get():.3f}")
        self.threshold_label.pack(side=tk.LEFT, padx=5)
        
        # Update threshold label when slider moves
        def update_threshold_label(*args):
            self.threshold_label.config(text=f"{self.threshold_value.get():.3f}")
        self.threshold_value.trace_add('write', update_threshold_label)
        
        # Count Threshold Input
        count_threshold_frame = ttk.Frame(control_frame)
        count_threshold_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(count_threshold_frame, text="Count Alert Threshold:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        self.count_threshold_entry = ttk.Entry(count_threshold_frame, textvariable=self.count_threshold, width=10)
        self.count_threshold_entry.pack(side=tk.LEFT, padx=5)
        self.count_threshold_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(count_threshold_frame, text="(Alert when count exceeds this value)").pack(side=tk.LEFT, padx=5)
        
        # ROI Controls
        roi_frame = ttk.LabelFrame(control_frame, text="Region of Interest", padding="5")
        roi_frame.pack(fill=tk.X, pady=5)
        
        self.select_roi_btn = ttk.Button(roi_frame, text="Select ROI", command=self.toggle_roi_selection)
        self.select_roi_btn.pack(side=tk.LEFT, padx=5)
        
        self.reset_roi_btn = ttk.Button(roi_frame, text="Reset ROI", command=self.reset_roi, state=tk.DISABLED)
        self.reset_roi_btn.pack(side=tk.LEFT, padx=5)
        
        # Control Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.start_btn = ttk.Button(button_frame, text="Start Processing", command=self.start_processing, width=20)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(button_frame, text="Stop", command=self.stop_processing, state=tk.DISABLED, width=15)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Exit", command=self.on_closing, width=15).pack(side=tk.LEFT, padx=5)
        
        # Alert Frame (dedicated space for alert)
        self.alert_frame = ttk.Frame(main_frame)
        self.alert_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Alert Label (Always visible but empty when inactive)
        self.alert_label = tk.Label(self.alert_frame, text="", 
                                     font=("Arial", 14, "bold"), 
                                     foreground="black", 
                                     background=self.root.cget("bg"),
                                     height=2) # Fixed height to reserve space
        self.alert_label.pack(fill=tk.X)

        self.canvas = None # Will be created in separate window
        
        # Status Bar
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_label = ttk.Label(status_frame, text="Status: Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.fps_label = ttk.Label(status_frame, text="FPS: -", relief=tk.SUNKEN, anchor=tk.E, width=15)
        self.fps_label.pack(side=tk.RIGHT)
        

        
    def on_input_change(self):
        choice = self.input_choice.get()
        if choice == "camera":
            self.file_entry.config(state=tk.DISABLED)
            self.browse_btn.config(state=tk.DISABLED)
            self.video_path.set("")
        else:
            self.file_entry.config(state=tk.NORMAL)
            self.browse_btn.config(state=tk.NORMAL)
    
    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
        )
        if filename:
            self.video_path.set(filename)
    
    def load_model(self):
        try:
            view = self.view_choice.get()
            
            if self.model_choice.get() == "csrnet":
                # Determine path based on view
                path = self.csrnet_path_front if view == "front" else self.csrnet_path_top
                
                # Check if we need to reload (simple check: if path changed or model not loaded)
                # For simplicity, we'll reload if the user requests it via Start. 
                # Ideally config changes should prob trigger reload state.
                
                self.update_status(f"Loading CSRNet ({view})...")
                
                # Check view to decide Batch Norm
                use_bn = True if view == "front" else False
                self.csrnet_model = CSRNet(batch_norm=use_bn).to(self.device)
                
                # Load weights handling both .pth (state_dict) and .tar (checkpoint dict)
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                
                state_dict = None
                if isinstance(checkpoint, dict):
                    if 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    elif 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    else:
                         # Fallback for dicts that are themselves the state_dict
                        # Check if values are tensors
                        is_pure_dict = True
                        for k, v in checkpoint.items():
                             if not isinstance(v, torch.Tensor):
                                 is_pure_dict = False
                                 break
                        if is_pure_dict:
                            state_dict = checkpoint
                
                if state_dict is None:
                     # Maybe it's not a dict, but the state_dict directly (unlikely)
                     state_dict = checkpoint
                
                # Strip 'module.' prefix if present
                if state_dict:
                    state_dict = self.strip_module_prefix(state_dict)
                    self.csrnet_model.load_state_dict(state_dict, strict=False) # strict=False to be safe with partial matches if needed
                
                self.csrnet_model.eval()
                self.update_status(f"CSRNet ({view}) loaded successfully")
                
            else:
                # YOLO
                path = self.yolo_path_front if view == "front" else self.yolo_path_top
                
                self.update_status(f"Loading YOLOv8 ({view})...")
                self.yolo_model = YOLO(path)
                self.update_status("YOLOv8 model loaded successfully")
            return True
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load model: {str(e)}")
            messagebox.showerror("Model Error", f"Failed to load model: {str(e)}")
            return False
    
    def create_video_window(self):
        if self.video_window is not None:
             try:
                 self.video_window.destroy()
             except:
                 pass
        
        self.video_window = tk.Toplevel(self.root)
        self.video_window.title("Video Output")
        self.video_window.geometry("1280x720")
        
        # Handle manual closing of video window
        self.video_window.protocol("WM_DELETE_WINDOW", self.stop_processing)
        
        self.canvas = tk.Canvas(self.video_window, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.image_item = None # Reset image item

    def start_processing(self):
        # Validate input
        if self.input_choice.get() == "file" and not self.video_path.get():
            messagebox.showwarning("Input Required", "Please select a video file")
            return
        
        # Load model
        if not self.load_model():
            return
        
        # Create video window
        self.create_video_window()
        
        # Open video source
        if self.input_choice.get() == "camera":
            try:
                idx = self.camera_index.get()
            except:
                idx = 0
            print(f"Attempting to open camera index: {idx}")
            self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.video_path.get())
        
        if not self.cap.isOpened():
            messagebox.showerror("Video Error", "Failed to open video source")
            return
        
        # Update UI
        self.is_processing = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_video, daemon=True)
        self.processing_thread.start()
    
    def stop_processing(self):
        self.is_processing = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        if self.cap:
            self.cap.release()
        self.update_status("Processing stopped")
        
        if self.video_window:
            try:
                self.video_window.destroy()
            except:
                pass
            self.video_window = None
            self.canvas = None
            self.image_item = None
    
    def resize_with_padding(self, image):
        old_height, old_width = image.shape[:2]
        scale = min(self.fixed_width / old_width, self.fixed_height / old_height)
        new_width = int(old_width * scale)
        new_height = int(old_height * scale)
        resized_img = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        pad_w = (self.fixed_width - new_width) // 2
        pad_h = (self.fixed_height - new_height) // 2
        padded_img = np.zeros((self.fixed_height, self.fixed_width, 3), dtype=image.dtype)
        padded_img[pad_h:pad_h+new_height, pad_w:pad_w+new_width, :] = resized_img
        return padded_img
    
        return overlay, count

    def process_csrnet(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_rgb = self.resize_with_padding(img_rgb)
        img_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.csrnet_model(img_tensor)
            density_map = output.squeeze().cpu().numpy()
        
        # Post-processing
        density_map = scipy.ndimage.median_filter(density_map, size=3)
        density_map = np.clip(density_map, 0, None)
        # Use threshold from slider
        threshold = self.threshold_value.get()
        density_map[density_map < threshold] = 0
        
        # Apply ROI Mask if active
        if self.roi_active and self.roi_coords:
            x1, y1, x2, y2 = self.roi_coords
            
            # CSRNet density map is smaller (1/8th size), so we must scale coordinates
            h_map, w_map = density_map.shape
            scale_x = w_map / self.fixed_width
            scale_y = h_map / self.fixed_height
            
            # specific scaling logic for CSRNet
            roi_x1 = int(x1 * scale_x)
            roi_x2 = int(x2 * scale_x)
            roi_y1 = int(y1 * scale_y)
            roi_y2 = int(y2 * scale_y)
            
            roi_x1 = max(0, roi_x1)
            roi_x2 = min(w_map, roi_x2)
            roi_y1 = max(0, roi_y1)
            roi_y2 = min(h_map, roi_y2)
            
            mask = np.zeros_like(density_map)
            mask[roi_y1:roi_y2, roi_x1:roi_x2] = 1
            density_map = density_map * mask
            
        count = np.sum(density_map)
        
        # Create visualization
        normalized = density_map.copy()
        if normalized.max() > 0:
            normalized = normalized / normalized.max()
        normalized = (normalized * 255).astype(np.uint8)
        normalized_resized = cv2.resize(normalized, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        heatmap = cv2.applyColorMap(normalized_resized, cv2.COLORMAP_JET)
        
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)
        
        # Draw ROI on overlay
        if self.roi_active and self.roi_coords:
             cv2.rectangle(overlay, (self.roi_coords[0], self.roi_coords[1]), (self.roi_coords[2], self.roi_coords[3]), (0, 255, 0), 2)
        
        # Add count text
        cv2.putText(overlay, f"Count: {count:.1f}", (30, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        
        return overlay, count
    
    def process_yolo(self, frame):
        frame = self.resize_with_padding(frame)
        results = self.yolo_model(frame, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy() if hasattr(results[0].boxes.xyxy, 'cpu') else results[0].boxes.xyxy
        
        # Filter boxes by ROI
        filtered_boxes = []
        if self.roi_active and self.roi_coords:
            roi_x1, roi_y1, roi_x2, roi_y2 = self.roi_coords
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                if roi_x1 <= cx <= roi_x2 and roi_y1 <= cy <= roi_y2:
                    filtered_boxes.append(box)
        else:
            filtered_boxes = boxes
            
        count = len(filtered_boxes)
        
        # Draw bounding boxes
        for box in filtered_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
        # Draw ROI
        if self.roi_active and self.roi_coords:
             cv2.rectangle(frame, (self.roi_coords[0], self.roi_coords[1]), (self.roi_coords[2], self.roi_coords[3]), (0, 255, 0), 2)
             
        # Add count text
        cv2.putText(frame, f"Count: {count}", (30, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
        
        return frame, count
    
    def process_video(self):
        import time
        frame_count = 0
        start_time = time.time()
        
        while self.is_processing:
            ret, frame = self.cap.read()
            if not ret:
                if self.input_choice.get() == "file":
                    # Video ended, loop back
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
            
            # Process frame based on model
            try:
                if self.model_choice.get() == "csrnet":
                    processed_frame, count = self.process_csrnet(frame)
                else:
                    processed_frame, count = self.process_yolo(frame)
                
                
                # Calculate FPS
                frame_count += 1
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                # Check count threshold and show alert if exceeded
                threshold = self.count_threshold.get()
                # print(f"Count: {count:.1f}, Threshold: {threshold}, Alert Active: {self.alert_active}")  # Debug
                if count > threshold:
                    if not self.alert_active:
                        self.root.after(0, self.show_alert)
                        self.alert_active = True
                else:
                    if self.alert_active:
                        self.root.after(0, self.hide_alert)
                        self.alert_active = False
                
                # Update display
                self.display_frame(processed_frame)
                self.root.after(0, self.update_status, f"Processing | Model: {self.model_choice.get().upper()} | Count: {count:.1f}")
                self.root.after(0, self.fps_label.config, {"text": f"FPS: {fps:.1f}"})
                
            except Exception as e:
                print(f"Processing error: {e}")
                continue
        
        if self.cap:
            self.cap.release()
        self.root.after(0, self.stop_btn.config, {"state": tk.DISABLED})
        self.root.after(0, self.start_btn.config, {"state": tk.NORMAL})
    
    def toggle_roi_selection(self):
        if not self.canvas:
             messagebox.showwarning("ROI Error", "Please start processing first to select ROI on the video.")
             return
             
        if self.roi_select_active:
            self.stop_roi_selection()
        else:
            self.start_roi_selection()

    def start_roi_selection(self):
        self.roi_select_active = True
        self.select_roi_btn.config(text="Cancel Selection")
        self.canvas.config(cursor="crosshair")
        
        # Bind mouse events
        self.canvas.bind("<ButtonPress-1>", self.on_roi_start)
        self.canvas.bind("<B1-Motion>", self.on_roi_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_roi_end)

    def stop_roi_selection(self):
        self.roi_select_active = False
        self.select_roi_btn.config(text="Select ROI")
        self.canvas.config(cursor="")
        self.canvas.unbind("<ButtonPress-1>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")
        self.roi_start = None
        self.roi_current = None

    def get_model_coords(self, canvas_x, canvas_y):
        """Convert canvas coordinates to model/image coordinates"""
        if self.display_scale == 0: return 0, 0
        
        model_x = int((canvas_x - self.display_offset_x) / self.display_scale)
        model_y = int((canvas_y - self.display_offset_y) / self.display_scale)
        
        # Clamp to image bounds
        model_x = max(0, min(model_x, self.fixed_width))
        model_y = max(0, min(model_y, self.fixed_height))
        
        return model_x, model_y

    def display_frame(self, frame):
        if not self.canvas: return
        
        # Initial frame is BGR
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Draw dynamic selection rectangle if selecting (in model coordinates)
        if self.roi_select_active and self.roi_start and self.roi_current:
             cv2.rectangle(frame_rgb, self.roi_start, self.roi_current, (0, 255, 255), 2)
        
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            img_h, img_w = frame_rgb.shape[:2]
            
            # Calculate scale to fit
            scale_w = canvas_width / img_w
            scale_h = canvas_height / img_h
            self.display_scale = min(scale_w, scale_h)
            
            new_w = int(img_w * self.display_scale)
            new_h = int(img_h * self.display_scale)
            
            # Resize for display
            img_pil = Image.fromarray(frame_rgb)
            img_pil = img_pil.resize((new_w, new_h), Image.Resampling.BILINEAR)
            
            # Calculate centering offsets
            self.display_offset_x = (canvas_width - new_w) // 2
            self.display_offset_y = (canvas_height - new_h) // 2
            
            imgtk = ImageTk.PhotoImage(image=img_pil)
            
            # Update canvas
            if self.image_item is None:
                self.image_item = self.canvas.create_image(
                    self.display_offset_x, self.display_offset_y, 
                    anchor=tk.NW, image=imgtk
                )
            else:
                self.canvas.coords(self.image_item, self.display_offset_x, self.display_offset_y)
                self.canvas.itemconfig(self.image_item, image=imgtk)
            self.canvas.image = imgtk
            
    def on_roi_start(self, event):
        mx, my = self.get_model_coords(event.x, event.y)
        self.roi_start = (mx, my)
        self.roi_current = (mx, my)

    def on_roi_drag(self, event):
        mx, my = self.get_model_coords(event.x, event.y)
        self.roi_current = (mx, my)

    def on_roi_end(self, event):
        if self.roi_start:
            x1, y1 = self.roi_start
            mx, my = self.get_model_coords(event.x, event.y)
            x2, y2 = mx, my
            
            # Ensure coords are top-left and bottom-right
            self.roi_coords = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            
            # Use ROI if area is significant
            if (self.roi_coords[2] - self.roi_coords[0]) > 10 and (self.roi_coords[3] - self.roi_coords[1]) > 10:
                self.activate_roi()
            
            self.stop_roi_selection()

    def activate_roi(self):
        self.roi_active = True
        self.reset_roi_btn.config(state=tk.NORMAL)
        self.scale_threshold()
        self.update_status(f"ROI Active: {self.roi_coords}")

    def reset_roi(self):
        self.roi_active = False
        self.roi_coords = None
        self.reset_roi_btn.config(state=tk.DISABLED)
        
        # Restore threshold
        if self.original_threshold is not None:
            self.count_threshold.set(self.original_threshold)
            self.original_threshold = None
            
        self.update_status("ROI Reset")

    def scale_threshold(self):
        if self.roi_coords:
            x1, y1, x2, y2 = self.roi_coords
            roi_area = (x2 - x1) * (y2 - y1)
            total_area = self.fixed_width * self.fixed_height
            ratio = roi_area / total_area
            
            # Store original if not already stored
            if self.original_threshold is None:
                self.original_threshold = self.count_threshold.get()
            
            new_threshold = int(self.original_threshold * ratio)
            self.count_threshold.set(max(1, new_threshold)) # Minimum 1
            print(f"Threshold scaled from {self.original_threshold} to {new_threshold} (Ratio: {ratio:.2f})")

    def update_status(self, message):
        self.status_label.config(text=f"Status: {message}")

    def show_alert(self):
        """Show the alert label when count threshold is exceeded"""
        try:
            if self.alert_label.cget("text") == "":
                self.alert_label.config(text="⚠️ ALERT: Count Threshold Exceeded! ⚠️")
                # Start flashing
                self.flash_alert()
                
                # Play sound
                if self.sound_enabled:
                    try:
                        if not pygame.mixer.music.get_busy():
                            pygame.mixer.music.load("beep.mp3")
                            pygame.mixer.music.play(-1) # Loop indefinitely
                    except Exception as e:
                        print(f"Error playing sound: {e}")
        except Exception as e:
            print(f"Error showing alert: {e}")

    def hide_alert(self):
        """Hide the alert label when count is below threshold"""
        try:
            if self.alert_label.cget("text") != "":
                self.alert_label.config(text="", background=self.root.cget("bg"), foreground="black")
                
                # Stop sound
                if self.sound_enabled:
                    pygame.mixer.music.stop()
        except Exception as e:
            print(f"Error hiding alert: {e}")

    def flash_alert(self):
        """Make the alert flash to get attention"""
        if self.alert_active:
            self.flash_state = not self.flash_state
            
            if self.flash_state:
                new_bg = "red"
                new_fg = "white"
            else:
                new_bg = "yellow"
                new_fg = "black"
            
            self.alert_label.config(background=new_bg, foreground=new_fg)
            self.root.after(500, self.flash_alert)  # Flash every 500ms

    def on_closing(self):
        if self.is_processing:
            self.stop_processing()
        if self.sound_enabled:
            pygame.mixer.quit()
        self.root.destroy()
        
if __name__ == "__main__":
    root = tk.Tk()
    app = CrowdCountingGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
