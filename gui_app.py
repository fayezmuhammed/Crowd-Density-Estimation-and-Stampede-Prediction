import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import customtkinter as ctk
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
import requests
import os
import time
import time
from dotenv import load_dotenv

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import sqlite3
import datetime
import csv
import pyttsx3 # For voice alerts

from flask import Flask, jsonify, send_file
import logging

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

load_dotenv()

ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class CrowdCountingGUI:
    def strip_module_prefix(self, state_dict):
        if any(k.startswith('module.') for k in state_dict.keys()):
            return {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        return state_dict

    def __init__(self, root):
        self.root = root
        self.root.title("CrowdSafe AI | Density Estimation & Monitoring")
        self.root.geometry("850x650") # Larger window for the new layout
        self.root.resizable(True, True)
        self.video_window = None # Toplevel window for video
        self.canvas = None # Video canvas object
        
        # Variables (We can still map Ctk to these TK variables)
        self.model_choice = ctk.StringVar(value="csrnet")
        self.view_choice = ctk.StringVar(value="front") # Front or Top view
        self.input_choice = ctk.StringVar(value="file")
        self.video_path = ctk.StringVar(value="")
        self.is_processing = False
        self.cap = None
        self.current_frame = None
        self.threshold_value = ctk.DoubleVar(value=0.022)  # Default CSRNet threshold
        self.count_threshold = ctk.IntVar(value=100)  # Default count threshold for alerts
        self.alert_active = False  # Track if alert is currently showing
        self.flash_state = False # Track flash state for color toggling
        self.camera_index = ctk.IntVar(value=0) # Camera index selection
        
        # Telegram Variables
        self.telegram_enabled = tk.BooleanVar(value=False)
        self.last_telegram_alert_time = 0
        self.telegram_cooldown = 60  # seconds between alerts
        
        # Smart Alert Variables
        self.sustained_warning_time = 0
        self.sustained_critical_time = 0
        self.alert_sustain_duration = ctk.DoubleVar(value=3.0) # Seconds to sustain before triggering
        
        self.roi_coords = None # (x1, y1, x2, y2)
        self.roi_active = False
        self.roi_select_active = False
        self.roi_start = None
        self.roi_current = None
        self.original_threshold = None # To store threshold before scaling
        self.image_item = None # Canvas image item ID
        
        # Analytics Variables
        self.history_timestamps = []
        self.history_counts = []
        self.dashboard_window = None
        self.dashboard_canvas = None
        self.dashboard_ax = None
        self.start_analytics_time = time.time()
        
        # Display Scaling Variables
        self.display_scale = 1.0
        self.display_offset_x = 0
        self.display_offset_y = 0

        # Calibration Variables
        self.calibration_active = False
        self.calibration_points = []


        
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

        # Initialize SQLite DB
        self.init_db()

        # Initialize Pyttsx3 for Voice Alerts
        try:
            self.tts_engine = pyttsx3.init()
            # Optional: set speech rate slightly faster
            rate = self.tts_engine.getProperty('rate')
            self.tts_engine.setProperty('rate', rate + 20)
            self.voice_enabled = True
        except Exception as e:
            print(f"TTS initialization failed: {e}")
            self.voice_enabled = False
            
        # Initialize Web API Server
        self.latest_count = None
        self.start_web_server()

        # Create GUI
        self.create_widgets()
        
    def init_db(self):
        """Initializes the SQLite database for historical recording."""
        try:
            self.conn = sqlite3.connect('history.db', check_same_thread=False)
            self.cursor = self.conn.cursor()
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS crowd_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    model TEXT,
                    count REAL
                )
            ''')
            self.conn.commit()
        except Exception as e:
            print(f"Database initialization error: {e}")

    def start_web_server(self):
        """Starts a lightweight Flask server on a daemon thread for external Tailscale access."""
        app = Flask(__name__)
        
        # Suppress Flask default logging in console
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

        @app.route('/')
        def serve_dashboard():
            html_path = os.path.join(os.path.dirname(__file__), 'web_dashboard.html')
            if os.path.exists(html_path):
                return send_file(html_path)
            return "Dashboard HTML file not found.", 404

        @app.route('/api/stats')
        def api_stats():
            # Gather state from Tkinter variables safely
            try:
                warn_val = self.warning_threshold.get()
                crit_val = self.count_threshold.get()
                model_val = self.model_choice.get().upper()
            except:
                warn_val, crit_val, model_val = 50, 100, "Unknown"

            data = {
                "current_count": self.latest_count,
                "is_processing": self.is_processing,
                "model": model_val,
                "thresholds": {
                    "warning": warn_val,
                    "critical": crit_val
                }
            }
            return jsonify(data)

        # Run Flask on 0.0.0.0 so Tailscale can route to it
        # Note: Threading is required so it doesn't block the Tkinter mainloop
        self.web_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False), daemon=True)
        self.web_thread.start()
        print("Web Dashboard running on http://0.0.0.0:5001 (accessible via Tailscale)")

    def create_widgets(self):
        # Configure grid layout (1 row, 2 columns)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # 1. Sidebar Frame
        self.sidebar_frame = ctk.CTkFrame(self.root, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        ctk.CTkLabel(self.sidebar_frame, text="CrowdSafe AI", font=ctk.CTkFont(size=20, weight="bold")).grid(row=0, column=0, padx=20, pady=(20, 10))
        
        self.start_btn = ctk.CTkButton(self.sidebar_frame, text="🚀 Start Processing", command=self.start_processing, 
                                       fg_color="#28a745", hover_color="#218838", font=ctk.CTkFont(weight="bold"))
        self.start_btn.grid(row=1, column=0, padx=20, pady=10)
        
        self.stop_btn = ctk.CTkButton(self.sidebar_frame, text="⏹️ Stop", command=self.stop_processing, 
                                      state=tk.DISABLED, fg_color="#dc3545", hover_color="#c82333")
        self.stop_btn.grid(row=2, column=0, padx=20, pady=10)

        self.analytics_btn = ctk.CTkButton(self.sidebar_frame, text="📊 Analytics", command=self.show_dashboard)
        self.analytics_btn.grid(row=3, column=0, padx=20, pady=10)

        # Appearance Mode Toggle
        self.appearance_mode_label = ctk.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=4, column=0, padx=20, pady=(20, 0))
        self.appearance_mode_optionemenu = ctk.CTkOptionMenu(self.sidebar_frame, values=["System", "Dark", "Light"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=5, column=0, padx=20, pady=(10, 20))

        ctk.CTkButton(self.sidebar_frame, text="🚪 Exit", command=self.on_closing, fg_color="transparent", border_width=2).grid(row=6, column=0, padx=20, pady=20)
        
        # 2. Main Content Frame
        self.content_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        self.content_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.content_frame.grid_columnconfigure(0, weight=1)
        self.content_frame.grid_rowconfigure(0, weight=1)

        # 3. Tabview for Settings
        self.tabview = ctk.CTkTabview(self.content_frame, width=600)
        self.tabview.grid(row=0, column=0, sticky="nsew")
        self.tabview.add("General")
        self.tabview.add("Advanced")
        self.tabview.add("Alerts & ROI")

        # --- Tab: General ---
        self.setup_general_tab()

        # --- Tab: Advanced ---
        self.setup_advanced_tab()

        # --- Tab: Alerts & ROI ---
        self.setup_alerts_tab()

        # 4. Alert Panel (Persistent at bottom of content)
        self.alert_frame = ctk.CTkFrame(self.content_frame, height=80, fg_color="transparent")
        self.alert_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        
        self.alert_label = ctk.CTkLabel(self.alert_frame, text="", font=ctk.CTkFont(size=16, weight="bold"), height=50)
        self.alert_label.pack(fill=tk.X, expand=True)

        # 5. Status Bar
        self.status_frame = ctk.CTkFrame(self.root, height=30, corner_radius=0)
        self.status_frame.grid(row=1, column=1, sticky="ew")
        
        self.status_label = ctk.CTkLabel(self.status_frame, text="Status: Ready", anchor="w")
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        self.fps_label = ctk.CTkLabel(self.status_frame, text="FPS: -", width=80)
        self.fps_label.pack(side=tk.RIGHT, padx=20)

    def setup_general_tab(self):
        tab = self.tabview.tab("General")
        tab.grid_columnconfigure(0, weight=1)

        # Model Selection
        model_group = ctk.CTkFrame(tab, fg_color="transparent")
        model_group.pack(fill=tk.X, pady=10, padx=20)
        ctk.CTkLabel(model_group, text="Inference Model", font=ctk.CTkFont(weight="bold")).pack(anchor="w")
        
        model_radio_frame = ctk.CTkFrame(model_group, fg_color="transparent")
        model_radio_frame.pack(fill=tk.X, pady=5)
        ctk.CTkRadioButton(model_radio_frame, text="CSRNet (High Precision Density)", variable=self.model_choice, value="csrnet").pack(side=tk.LEFT, padx=(0, 20))
        ctk.CTkRadioButton(model_radio_frame, text="YOLOv8 (Fast Object Detection)", variable=self.model_choice, value="yolo").pack(side=tk.LEFT)

        # View Selection
        view_group = ctk.CTkFrame(tab, fg_color="transparent")
        view_group.pack(fill=tk.X, pady=10, padx=20)
        ctk.CTkLabel(view_group, text="Camera Perspective", font=ctk.CTkFont(weight="bold")).pack(anchor="w")
        
        view_radio_frame = ctk.CTkFrame(view_group, fg_color="transparent")
        view_radio_frame.pack(fill=tk.X, pady=5)
        ctk.CTkRadioButton(view_radio_frame, text="Front/Wide View", variable=self.view_choice, value="front").pack(side=tk.LEFT, padx=(0, 20))
        ctk.CTkRadioButton(view_radio_frame, text="Top/Aereal View", variable=self.view_choice, value="top").pack(side=tk.LEFT)

        # Input Selection
        input_group = ctk.CTkFrame(tab, fg_color="transparent")
        input_group.pack(fill=tk.X, pady=10, padx=20)
        ctk.CTkLabel(input_group, text="Video Source", font=ctk.CTkFont(weight="bold")).pack(anchor="w")
        
        input_radio_frame = ctk.CTkFrame(input_group, fg_color="transparent")
        input_radio_frame.pack(fill=tk.X, pady=5)
        ctk.CTkRadioButton(input_radio_frame, text="Local File", variable=self.input_choice, value="file", command=self.on_input_change).pack(side=tk.LEFT, padx=(0, 20))
        ctk.CTkRadioButton(input_radio_frame, text="Live Camera", variable=self.input_choice, value="camera", command=self.on_input_change).pack(side=tk.LEFT)
        
        ctk.CTkLabel(input_radio_frame, text="Camera Index:").pack(side=tk.LEFT, padx=(20, 5))
        self.camera_entry = ctk.CTkEntry(input_radio_frame, textvariable=self.camera_index, width=50)
        self.camera_entry.pack(side=tk.LEFT)

        # File Selection
        self.file_frame_content = ctk.CTkFrame(tab, fg_color="transparent")
        self.file_frame_content.pack(fill=tk.X, pady=10, padx=20)
        self.file_entry = ctk.CTkEntry(self.file_frame_content, textvariable=self.video_path, placeholder_text="Path to video file...")
        self.file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.browse_btn = ctk.CTkButton(self.file_frame_content, text="📁 Browse", width=100, command=self.browse_file)
        self.browse_btn.pack(side=tk.LEFT)

    def setup_advanced_tab(self):
        tab = self.tabview.tab("Advanced")
        
        # Original Precision Group
        prec_group = ctk.CTkFrame(tab, fg_color="transparent")
        prec_group.pack(fill=tk.X, pady=10, padx=20)
        ctk.CTkLabel(prec_group, text="Density Sensitivity (CSRNet)", font=ctk.CTkFont(weight="bold")).pack(anchor="w")
        
        slider_row = ctk.CTkFrame(prec_group, fg_color="transparent")
        slider_row.pack(fill=tk.X, pady=5)
        self.thresh_slider = ctk.CTkSlider(slider_row, from_=0.001, to=0.100, variable=self.threshold_value)
        self.thresh_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.threshold_label = ctk.CTkLabel(slider_row, text=f"{self.threshold_value.get():.3f}", width=50)
        self.threshold_label.pack(side=tk.LEFT)
        self.thresh_slider.configure(command=lambda val: self.threshold_label.configure(text=f"{float(val):.3f}"))
        ctk.CTkLabel(prec_group, text="Adjusts how strictly the CSRNet model counts dark/noisy areas.", font=ctk.CTkFont(size=11), text_color="gray").pack(anchor="w")

        # New Vision Analytics Group
        visual_group = ctk.CTkFrame(tab, fg_color="transparent")
        visual_group.pack(fill=tk.X, pady=(20, 10), padx=20)
        ctk.CTkLabel(visual_group, text="Live Vision Analytics", font=ctk.CTkFont(weight="bold")).pack(anchor="w")
        
        v_row1 = ctk.CTkFrame(visual_group, fg_color="transparent")
        v_row1.pack(fill=tk.X, pady=5)
        
        self.enable_blur = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(v_row1, text="🌫️ Privacy Mode (Blur Faces)", variable=self.enable_blur).pack(side=tk.LEFT, padx=(0, 20))
        
        self.enable_heatmap = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(v_row1, text="🔥 Density Heatmap Overlay", variable=self.enable_heatmap).pack(side=tk.LEFT, padx=(0, 20))
        
        v_row2 = ctk.CTkFrame(visual_group, fg_color="transparent")
        v_row2.pack(fill=tk.X, pady=5)
        
        self.enable_social_distancing = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(v_row2, text="📏 Social Distancing Warning (YOLO Only)", variable=self.enable_social_distancing).pack(side=tk.LEFT)

        # Calibration
        cal_group = ctk.CTkFrame(tab, fg_color="transparent")
        cal_group.pack(fill=tk.X, pady=20, padx=20)
        ctk.CTkLabel(cal_group, text="Distance Calibration", font=ctk.CTkFont(weight="bold")).pack(anchor="w")
        ctk.CTkLabel(cal_group, text="Define a 1-meter reference on the video for accurate area calculations.", font=ctk.CTkFont(size=11), text_color="gray").pack(anchor="w")
        
        self.cal_btn = ctk.CTkButton(cal_group, text="📏 Calibrate 1m", command=self.toggle_calibration, width=200)
        self.cal_btn.pack(pady=10, anchor="w")

    def setup_alerts_tab(self):
        tab = self.tabview.tab("Alerts & ROI")

        # Count Thresholds
        alert_group = ctk.CTkFrame(tab, fg_color="transparent")
        alert_group.pack(fill=tk.X, pady=10, padx=20)
        ctk.CTkLabel(alert_group, text="Crowd Limit Thresholds", font=ctk.CTkFont(weight="bold")).pack(anchor="w")
        
        limit_row = ctk.CTkFrame(alert_group, fg_color="transparent")
        limit_row.pack(fill=tk.X, pady=5)
        
        # Warning Threshold
        ctk.CTkLabel(limit_row, text="Warning Level:").pack(side=tk.LEFT, padx=(0, 5))
        self.warning_threshold = ctk.IntVar(value=50)
        self.warning_entry = ctk.CTkEntry(limit_row, textvariable=self.warning_threshold, width=60)
        self.warning_entry.pack(side=tk.LEFT, padx=(0, 20))
        
        # Critical Threshold
        ctk.CTkLabel(limit_row, text="Critical Level:").pack(side=tk.LEFT, padx=(0, 5))
        self.count_threshold_entry = ctk.CTkEntry(limit_row, textvariable=self.count_threshold, width=60)
        self.count_threshold_entry.pack(side=tk.LEFT, padx=(0, 10))

        # Smart Alerts Timing
        time_row = ctk.CTkFrame(alert_group, fg_color="transparent")
        time_row.pack(fill=tk.X, pady=5)
        ctk.CTkLabel(time_row, text="Sustain Duration (sec):").pack(side=tk.LEFT, padx=(0, 5))
        self.sustain_entry = ctk.CTkEntry(time_row, textvariable=self.alert_sustain_duration, width=60)
        self.sustain_entry.pack(side=tk.LEFT, padx=(0, 20))
        ctk.CTkLabel(time_row, text="Delay before alert triggers (prevents false alarms)").pack(side=tk.LEFT)

        # Telegram
        tg_group = ctk.CTkFrame(tab, fg_color="transparent")
        tg_group.pack(fill=tk.X, pady=10, padx=20)
        ctk.CTkLabel(tg_group, text="External Notifications", font=ctk.CTkFont(weight="bold")).pack(anchor="w")
        
        tg_row = ctk.CTkFrame(tg_group, fg_color="transparent")
        tg_row.pack(fill=tk.X, pady=5)
        ctk.CTkCheckBox(tg_row, text="Enable 📱 Telegram Alerts", variable=self.telegram_enabled).pack(side=tk.LEFT, padx=(0, 20))
        
        # Voice Alerts Toggle
        self.voice_alerts_enabled = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(tg_row, text="🎤 Voice Alerts", variable=self.voice_alerts_enabled).pack(side=tk.LEFT, padx=(0, 20))
        
        self.test_telegram_btn = ctk.CTkButton(tg_row, text="Test Send", command=self.test_telegram_alert, width=100)
        self.test_telegram_btn.pack(side=tk.LEFT)

        # ROI
        roi_group = ctk.CTkFrame(tab, fg_color="transparent")
        roi_group.pack(fill=tk.X, pady=10, padx=20)
        ctk.CTkLabel(roi_group, text="Region of Interest (ROI)", font=ctk.CTkFont(weight="bold")).pack(anchor="w")
        ctk.CTkLabel(roi_group, text="Focus monitoring on a specific sub-area of the camera feed.", font=ctk.CTkFont(size=11), text_color="gray").pack(anchor="w")
        
        roi_row = ctk.CTkFrame(roi_group, fg_color="transparent")
        roi_row.pack(fill=tk.X, pady=10)
        self.select_roi_btn = ctk.CTkButton(roi_row, text="🎯 Select ROI", command=self.toggle_roi_selection, width=150)
        self.select_roi_btn.pack(side=tk.LEFT, padx=(0, 10))
        self.reset_roi_btn = ctk.CTkButton(roi_row, text="🔄 Reset", command=self.reset_roi, state=tk.DISABLED, width=100)
        self.reset_roi_btn.pack(side=tk.LEFT)        

    def change_appearance_mode_event(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)
        
    def on_input_change(self):
        choice = self.input_choice.get()
        if choice == "camera":
            self.file_entry.configure(state=tk.DISABLED)
            self.browse_btn.configure(state=tk.DISABLED)
            self.video_path.set("")
        else:
            self.file_entry.configure(state=tk.NORMAL)
            self.browse_btn.configure(state=tk.NORMAL)
    
    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
        )
        if filename:
            self.video_path.set(filename)
    
    def set_controls_state(self, state):
        """Helper to disable/enable settings tabs during processing"""
        self.tabview.configure(state=state)
        # Explicitly disable inputs that might still be clickable
        self.browse_btn.configure(state=state)
        self.select_roi_btn.configure(state=state)
        self.cal_btn.configure(state=state)

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
        
        self.video_window = ctk.CTkToplevel(self.root)
        self.video_window.title("Video Output")
        self.video_window.geometry("1024x720")
        
        # Handle manual closing of video window
        self.video_window.protocol("WM_DELETE_WINDOW", self.stop_processing)
        
        # ctk doesn't have CTkCanvas, relying on standard tk.Canvas works fine for image displaying inside Toplevel
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
            import platform
            if platform.system() == "Darwin":
                self.cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
            elif platform.system() == "Windows":
                self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(idx)
        else:
            self.cap = cv2.VideoCapture(self.video_path.get())
        
        if not self.cap.isOpened():
            messagebox.showerror("Video Error", "Failed to open video source")
            return
        
        # Update UI
        self.is_processing = True
        self.start_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        
        # Disable settings during processing
        self.set_controls_state(tk.DISABLED)
        
        # Start processing thread
        
        self._thread_model_choice = self.model_choice.get()
        self._thread_input_choice = self.input_choice.get()
        
        self.processing_thread = threading.Thread(target=self.process_video, daemon=True)
        self.processing_thread.start()
    
    def stop_processing(self):
        self.is_processing = False
        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.set_controls_state(tk.NORMAL)
        self.update_status("Processing stopped")
        # Cleanup is handled by the processing thread when it exits
    
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
        img_padded = self.resize_with_padding(img_rgb)
        img_tensor = self.transform(img_padded).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.csrnet_model(img_tensor)
            density_map = output.squeeze().cpu().numpy()
        
        # Post-processing
        density_map = scipy.ndimage.median_filter(density_map, size=3)
        density_map = np.clip(density_map, 0, None)
        
        # Apply ROI Mask if active
        if self.roi_active and self.roi_coords:
            x1, y1, x2, y2 = self.roi_coords
            
            # CSRNet density map is smaller (1/8th size), so we must scale coordinates
            h_map, w_map = density_map.shape
            scale_x = w_map / self.fixed_width
            scale_y = h_map / self.fixed_height
            
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
            
        # Use threshold from slider
        threshold = self.threshold_value.get()
        density_map[density_map < threshold] = 0
        
        count = np.sum(density_map)
        
        # Create visualization
        overlay = cv2.cvtColor(img_padded, cv2.COLOR_RGB2BGR) # Start with padded BGR frame
        
        # Optional Heatmap Overlay
        if self.enable_heatmap.get():
            normalized = density_map.copy()
            if normalized.max() > 0:
                normalized = normalized / normalized.max()
            normalized = (normalized * 255).astype(np.uint8)
            normalized_resized = cv2.resize(normalized, (img_padded.shape[1], img_padded.shape[0]), interpolation=cv2.INTER_LINEAR)
            heatmap = cv2.applyColorMap(normalized_resized, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(overlay, 0.6, heatmap, 0.4, 0)
        
        # Draw ROI on overlay
        if self.roi_active and self.roi_coords:
             cv2.rectangle(overlay, (self.roi_coords[0], self.roi_coords[1]), (self.roi_coords[2], self.roi_coords[3]), (0, 255, 0), 2)
        
        # Add count text
        cv2.putText(overlay, f"Count: {count:.1f}", (30, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        
        return overlay, count
    
    def process_yolo(self, frame):
        frame_padded = self.resize_with_padding(frame)
        # YOLO Inference
        results = self.yolo_model(frame_padded, classes=0, conf=0.3, verbose=False) # classes=0 for 'person'
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        filtered_boxes = []
        # Filter by ROI if active
        if self.roi_active and self.roi_coords:
            x_min, y_min, x_max, y_max = self.roi_coords
            for box in boxes:
                x1, y1, x2, y2 = box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                if (x_min <= center_x <= x_max) and (y_min <= center_y <= y_max):
                    filtered_boxes.append(box)
        else:
             filtered_boxes = boxes
             
        count = len(filtered_boxes)
        
        # Social Distancing check (if model is calibrated)
        if self.enable_social_distancing.get() and len(self.calibration_points) == 2:
            p1, p2 = self.calibration_points
            pixel_distance_1m = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            
            # Ensure pixel_distance_1m is not zero to avoid division by zero
            if pixel_distance_1m > 0:
                safe_distance_pixels = pixel_distance_1m * 1.5 # 1.5 meters safe distance
                
                centers = []
                for box in filtered_boxes:
                    centers.append(((box[0] + box[2])/2, (box[1] + box[3])/2))
                    
                # Naive O(n^2) distance check for demonstration
                for i in range(len(centers)):
                    for j in range(i+1, len(centers)):
                        dist = np.sqrt((centers[i][0]-centers[j][0])**2 + (centers[i][1]-centers[j][1])**2)
                        if dist < safe_distance_pixels:
                            # Draw warning line
                            cv2.line(frame_padded, (int(centers[i][0]), int(centers[i][1])), 
                                     (int(centers[j][0]), int(centers[j][1])), (0, 0, 255), 2)
        
        # Draw bounding boxes (and apply blur)
        for box in filtered_boxes:
            x1, y1, x2, y2 = map(int, box)
            
            # Apply PII Privacy Blurring (Faces) if enabled
            if getattr(self, "enable_blur", None) and self.enable_blur.get():
                head_y2 = y1 + int((y2 - y1) * 0.3) # Top 30% of box
                # Apply blur safely within frame bounds
                if y1 >= 0 and head_y2 < frame_padded.shape[0] and x1 >= 0 and x2 < frame_padded.shape[1]:
                    roi = frame_padded[y1:head_y2, x1:x2]
                    if roi.size > 0:
                        # Ensure kernel size is odd and >= 3
                        kw, kh = max(3, (x2-x1)//2), max(3, (head_y2-y1)//2)
                        kw = kw if kw % 2 != 0 else kw + 1
                        kh = kh if kh % 2 != 0 else kh + 1
                        blur = cv2.GaussianBlur(roi, (kw, kh), 0)
                        frame_padded[y1:head_y2, x1:x2] = blur
            
            cv2.rectangle(frame_padded, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
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
                if self._thread_input_choice == "file":
                    # Video ended, loop back
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
            
            # Process frame based on model
            try:
                if self._thread_model_choice == "csrnet":
                    processed_frame, count = self.process_csrnet(frame)
                else:
                    processed_frame, count = self.process_yolo(frame)
                
                
                # Calculate FPS
                frame_count += 1
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                # Record data to DB
                try:
                    if hasattr(self, 'cursor'):
                        self.cursor.execute("INSERT INTO crowd_data (model, count) VALUES (?, ?)", 
                                            (self._thread_model_choice, float(count)))
                        self.conn.commit()
                except Exception as e:
                    print(f"DB Write Error: {e}")
                    
                # Update latest count for Web API
                self.latest_count = count
                    
                # Update Analytics Data (in-memory for live chart)
                current_time = time.time() - self.start_analytics_time
                self.history_timestamps.append(current_time)
                self.history_counts.append(count)
                
                if len(self.history_timestamps) > 100:
                    self.history_timestamps.pop(0)
                    self.history_counts.pop(0)
                
                # Update display on main thread (tkinter is not thread-safe)
                frame_copy = processed_frame.copy()
                current_model_txt = self._thread_model_choice.upper()
                self.root.after(0, self.update_main_gui, frame_copy, current_model_txt, count, fps)
                
                # Small delay to prevent flooding the event queue
                time.sleep(0.01)
                
            except Exception as e:
                print(f"Processing error: {e}")
                continue
        
        # Cleanup (runs on processing thread after loop exits)
        if self.cap:
            self.cap.release()
            self.cap = None
        
        def _cleanup_ui():
            self.stop_btn.configure(state=tk.DISABLED)
            self.start_btn.configure(state=tk.NORMAL)
            if self.video_window:
                try:
                    self.video_window.destroy()
                except:
                    pass
                self.video_window = None
                self.canvas = None
                self.image_item = None
        
        self.root.after(0, _cleanup_ui)
    
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
        self.select_roi_btn.configure(text="Cancel Selection")
        self.canvas.config(cursor="crosshair")
        
        # Bind mouse events
        self.canvas.bind("<ButtonPress-1>", self.on_roi_start)
        self.canvas.bind("<B1-Motion>", self.on_roi_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_roi_end)

    def stop_roi_selection(self):
        self.roi_select_active = False
        self.select_roi_btn.configure(text="Select ROI")
        self.canvas.config(cursor="")
        self.canvas.unbind("<ButtonPress-1>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")
        self.roi_start = None
        self.roi_current = None

    # --- Calibration Logic ---
    def toggle_calibration(self):
        if not self.canvas:
             messagebox.showwarning("Calibration Error", "Please start processing first.")
             return
             
        if self.calibration_active:
            self.stop_calibration()
        else:
            self.start_calibration()
            
    def start_calibration(self):
        self.calibration_active = True
        self.calibration_points = []
        self.cal_btn.configure(text="Cancel Calib")
        
        # Configure canvas cursor
        try:
             self.canvas.config(cursor="tcross")
        except:
             self.canvas.config(cursor="crosshair") # Fallback
             
        self.update_status("Click two points on the video that act as 1 meter.")
        
        # Unbind ROI events to avoid conflict
        if self.roi_select_active:
            self.stop_roi_selection()
            
        self.canvas.bind("<ButtonPress-1>", self.on_calibration_click)
        
    def stop_calibration(self):
        self.calibration_active = False
        self.cal_btn.configure(text="Calibrate 1m")
        self.canvas.config(cursor="")
        self.canvas.unbind("<ButtonPress-1>")
        self.calibration_points = []
        
    def on_calibration_click(self, event):
        mx, my = self.get_model_coords(event.x, event.y)
        
        # Draw point on canvas
        r = 3
        self.canvas.create_oval(event.x-r, event.y-r, event.x+r, event.y+r, fill="red", outline="white", tags="calib_mark")
        
        self.calibration_points.append((mx, my))
        
        if len(self.calibration_points) == 2:
            p1 = self.calibration_points[0]
            p2 = self.calibration_points[1]
            
            # Calculate distance in pixels
            dist_px = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            
            if dist_px < 5:
                messagebox.showwarning("Error", "Points are too close.")
                self.start_calibration() # Restart
                return
                
            # Logic: dist_px corresponds to 1 meter.
            px_per_meter = dist_px
            
            # Calculate total area of view in meters
            # Width_m * Height_m
            width_m = self.fixed_width / px_per_meter
            height_m = self.fixed_height / px_per_meter
            area_m2 = width_m * height_m
            
            # Critical Density constant: 4 people / m^2
            CRITICAL_DENSITY = 4 
            suggested_threshold = int(area_m2 * CRITICAL_DENSITY)
            
            response = messagebox.askyesno(
                "Calibration Complete", 
                f"Pixels per meter: {dist_px:.1f} px\n"
                f"Est. Total Area: {area_m2:.1f} m²\n\n"
                f"Calculated Safe Threshold: {suggested_threshold}\n"
                f"(Based on 4 ppl/m²)\n\n"
                f"Apply this threshold?"
            )
            
            if response:
                self.count_threshold.set(suggested_threshold)
                self.update_status(f"Threshold set to {suggested_threshold}")
            
            # Clean up
            self.canvas.delete("calib_mark")
            self.stop_calibration()

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
        self.reset_roi_btn.configure(state=tk.NORMAL)
        self.scale_threshold()
        self.update_status(f"ROI Active: {self.roi_coords}")

    def reset_roi(self):
        self.roi_active = False
        self.roi_coords = None
        self.reset_roi_btn.configure(state=tk.DISABLED)
        
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
        self.status_label.configure(text=f"Status: {message}")

    def update_main_gui(self, frame_copy, current_model_txt, count, fps):
        """Batched GUI updates to prevent Tkinter event starvation"""
        self.display_frame(frame_copy)
        self.update_status(f"Processing | Model: {current_model_txt} | Count: {count:.1f}")
        self.fps_label.configure(text=f"FPS: {fps:.1f}")

    def send_telegram_alert(self, message):
        if not self.telegram_enabled.get():
            return
            
        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        
        if not token or not chat_id:
            print("Telegram alert skipped: Token or Chat ID not configured in .env file.")
            messagebox.showwarning("Telegram Error", "Please configure TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in your .env file.")
            self.telegram_enabled.set(False)
            return

        def _send():
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            payload = {
                "chat_id": chat_id,
                "text": message
            }
            try:
                response = requests.post(url, json=payload, timeout=5)
                if response.status_code != 200:
                    print(f"Failed to send Telegram alert: {response.text}")
            except Exception as e:
                print(f"Error sending Telegram alert: {e}")
        
        # Run in background to avoid blocking GUI
        threading.Thread(target=_send, daemon=True).start()

    def test_telegram_alert(self):
        self.send_telegram_alert("🧪 Test Alert from Crowd Counting System 🧪\n\nIf you are seeing this, your Telegram configuration is correct!")
        messagebox.showinfo("Telegram Test", "Test alert dispatched! Check your Telegram app.")

    def check_alerts(self, count):
        """Checks thresholds and triggers appropriate alerts based on logic"""
        warning_level = self.warning_threshold.get()
        critical_level = self.count_threshold.get()
        sustain_req = self.alert_sustain_duration.get()
        current_time = time.time()

        # Handle Critical Logic
        if count >= critical_level:
            if self.sustained_critical_time == 0:
                self.sustained_critical_time = current_time # Start tracking
            elif current_time - self.sustained_critical_time >= sustain_req:
                self.show_alert(count, level="CRITICAL")
        else:
            self.sustained_critical_time = 0 # Reset if count drops

        # Handle Warning Logic (only if not critical)
        if count >= warning_level and count < critical_level:
            if self.sustained_warning_time == 0:
                self.sustained_warning_time = current_time
            elif current_time - self.sustained_warning_time >= sustain_req:
                self.show_alert(count, level="WARNING")
        else:
            self.sustained_warning_time = 0 # Reset
            
        # Hide if completely normal
        if count < warning_level and count < critical_level:
            self.hide_alert()

    def show_alert(self, count=None, level="CRITICAL"):
        """Show the alert label with appropriate severity level"""
        try:
            current_text = self.alert_label.cget("text")
            
            if level == "CRITICAL" and "CRITICAL" not in current_text:
                self.alert_label.configure(text=f"🚨 CRITICAL: Crowd Limit Exceeded! (Count: {int(count)}) 🚨")
                self.alert_active = True
                self.flash_alert()
                
                # Check Telegram Cooldown and Dispatch
                current_time = time.time()
                if current_time - self.last_telegram_alert_time >= self.telegram_cooldown:
                    count_str = f"{count:.1f}" if count is not None else "Unknown"
                    msg = f"🚨 CRITICAL CROWD ALERT! 🚨\n\nThe crowd count threshold has been exceeded!\nCurrent Count: {count_str}\nThreshold: {self.count_threshold.get()}"
                    self.send_telegram_alert(msg)
                    self.last_telegram_alert_time = current_time
                    
                # Action Voice TTS locally if enabled
                if getattr(self, "voice_enabled", False) and getattr(self, "voice_alerts_enabled", None) and self.voice_alerts_enabled.get():
                    def speak_alert():
                        try:
                            self.tts_engine.say(f"Critical Alert! Crowd count has reached {int(count)}")
                            self.tts_engine.runAndWait()
                        except:
                            pass
                    threading.Thread(target=speak_alert, daemon=True).start()
                
                # Play sound
                if self.sound_enabled:
                    try:
                        if not pygame.mixer.music.get_busy():
                            pygame.mixer.music.load("beep.mp3")
                            pygame.mixer.music.play(-1) # Loop indefinitely
                    except Exception as e:
                        print(f"Error playing sound: {e}")
                        
            elif level == "WARNING" and "WARNING" not in current_text:
                self.alert_label.configure(text=f"⚠️ WARNING: High Traffic Detected (Count: {int(count)})", fg_color="orange", text_color="black")
                self.alert_active = False # Stop flashing for warnings
                self.alert_label.update_idletasks()
                
                if self.sound_enabled:
                     pygame.mixer.music.stop()

        except Exception as e:
            print(f"Error showing alert: {e}")

    def hide_alert(self):
        """Hide the alert label when count is below threshold"""
        try:
            if self.alert_label.cget("text") != "":
                self.alert_label.configure(text="", fg_color="transparent", text_color="black")
                self.alert_label.update_idletasks()
                
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
            
            self.alert_label.configure(fg_color=new_bg, text_color=new_fg)
            self.alert_label.update_idletasks()
            self.root.after(500, self.flash_alert)  # Flash every 500ms

    def show_dashboard(self):
        """Creates or brings to front the live analytics dashboard."""
        if self.dashboard_window is None or not self.dashboard_window.winfo_exists():
            self.dashboard_window = ctk.CTkToplevel(self.root)
            self.dashboard_window.title("Analytics Dashboard")
            self.dashboard_window.geometry("600x450")
            
            # Top Controls Frame
            ctrl_frame = ctk.CTkFrame(self.dashboard_window, fg_color="transparent")
            ctrl_frame.pack(fill=tk.X, padx=10, pady=(10, 0))
            
            self.export_btn = ctk.CTkButton(ctrl_frame, text="📥 Export CSV", command=self.export_csv, width=120)
            self.export_btn.pack(side=tk.RIGHT)
            
            # Setup Matplotlib Figure
            fig = Figure(figsize=(6, 4), dpi=100, facecolor="#2b2b2b")
            self.dashboard_ax = fig.add_subplot(111)
            self.dashboard_ax.set_facecolor("#2b2b2b")
            self.dashboard_ax.tick_params(colors="white")
            self.dashboard_ax.xaxis.label.set_color("white")
            self.dashboard_ax.yaxis.label.set_color("white")
            self.dashboard_ax.title.set_color("white")
            
            self.dashboard_canvas = FigureCanvasTkAgg(fig, master=self.dashboard_window)
            self.dashboard_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Initial draw
            self.update_dashboard()
        else:
            self.dashboard_window.focus()

    def update_dashboard(self):
        """Updates the plot with the latest data if the window is open."""
        if self.dashboard_ax is not None and self.dashboard_canvas is not None:
            self.dashboard_ax.clear()
            self.dashboard_ax.plot(self.history_timestamps, self.history_counts, color="#00e676", linewidth=2)
            self.dashboard_ax.set_title("Live Crowd Density", color="white", pad=10)
            self.dashboard_ax.set_xlabel("Time (s)", color="white")
            self.dashboard_ax.set_ylabel("Crowd Count", color="white")
            
            # Threshold line
            if self.count_threshold.get() > 0:
                self.dashboard_ax.axhline(y=self.count_threshold.get(), color='r', linestyle='--', linewidth=1, label="Threshold")
                self.dashboard_ax.legend(loc="upper right")
                
            self.dashboard_canvas.draw()

    def export_csv(self):
        """Exports the SQLite historical data to a CSV file."""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                initialfile=f"crowd_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
                title="Save Crowd Report"
            )
            
            if filename:
                self.cursor.execute("SELECT timestamp, model, count FROM crowd_data ORDER BY timestamp DESC")
                rows = self.cursor.fetchall()
                
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Timestamp", "Model", "Density Count"])
                    writer.writerows(rows)
                messagebox.showinfo("Export Successful", f"Data exported to:\n{filename}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data: {e}")

    def on_closing(self):
        if self.is_processing:
            self.stop_processing()
        if self.sound_enabled:
            pygame.mixer.quit()
        if hasattr(self, 'conn'):
            self.conn.close()
        self.root.destroy()
        
if __name__ == "__main__":
    root = ctk.CTk()
    app = CrowdCountingGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
