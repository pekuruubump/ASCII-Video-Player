import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import time
from PIL import Image, ImageTk, ImageDraw, ImageFont
import os

class ASCIIVideoGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ASCII Video Player 🎥")
        self.root.geometry("1000x700")
        
        # Variables
        self.video_path = None
        self.cap = None
        self.playing = False
        self.current_frame = None
        self.ascii_width = 100
        self.font_size = 10
        self.chars = " .:-=+*#%@"
        
        # Color scheme
        self.bg_color = "#1e1e1e"
        self.fg_color = "#ffffff"
        self.accent_color = "#007acc"
        
        self.root.configure(bg=self.bg_color)
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main container
        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = tk.Frame(main_frame, bg=self.bg_color)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # File selection
        file_frame = tk.Frame(control_frame, bg=self.bg_color)
        file_frame.pack(side=tk.LEFT, padx=5)
        
        self.file_label = tk.Label(file_frame, text="No file selected", 
                                   bg=self.bg_color, fg="#888888",
                                   font=("Arial", 10))
        self.file_label.pack(side=tk.LEFT, padx=5)
        
        self.btn_open = tk.Button(file_frame, text="📁 Open Video", 
                                  command=self.open_video,
                                  bg=self.accent_color, fg="white",
                                  relief=tk.FLAT, padx=10)
        self.btn_open.pack(side=tk.LEFT, padx=5)
        
        # Playback controls
        playback_frame = tk.Frame(control_frame, bg=self.bg_color)
        playback_frame.pack(side=tk.LEFT, padx=20)
        
        self.btn_play = tk.Button(playback_frame, text="▶ Play", 
                                  command=self.toggle_playback,
                                  bg="#2d2d2d", fg="white",
                                  relief=tk.FLAT, padx=10, state=tk.DISABLED)
        self.btn_play.pack(side=tk.LEFT, padx=2)
        
        self.btn_stop = tk.Button(playback_frame, text="⏹ Stop", 
                                  command=self.stop_playback,
                                  bg="#2d2d2d", fg="white",
                                  relief=tk.FLAT, padx=10, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=2)
        
        # Settings
        settings_frame = tk.Frame(control_frame, bg=self.bg_color)
        settings_frame.pack(side=tk.RIGHT, padx=5)
        
        tk.Label(settings_frame, text="Width:", bg=self.bg_color, fg=self.fg_color).pack(side=tk.LEFT, padx=2)
        self.width_var = tk.StringVar(value="100")
        width_entry = tk.Entry(settings_frame, textvariable=self.width_var, width=5,
                              bg="#2d2d2d", fg=self.fg_color, relief=tk.FLAT)
        width_entry.pack(side=tk.LEFT, padx=2)
        
        tk.Label(settings_frame, text="Font size:", bg=self.bg_color, fg=self.fg_color).pack(side=tk.LEFT, padx=2)
        self.font_var = tk.StringVar(value="10")
        font_entry = tk.Entry(settings_frame, textvariable=self.font_var, width=5,
                             bg="#2d2d2d", fg=self.fg_color, relief=tk.FLAT)
        font_entry.pack(side=tk.LEFT, padx=2)
        
        self.btn_apply = tk.Button(settings_frame, text="Apply", 
                                   command=self.apply_settings,
                                   bg="#2d2d2d", fg="white",
                                   relief=tk.FLAT, padx=10)
        self.btn_apply.pack(side=tk.LEFT, padx=5)
        
        # ASCII Display area
        display_frame = tk.Frame(main_frame, bg="#000000")
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas for ASCII art
        self.canvas = tk.Canvas(display_frame, bg="#000000", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbars
        v_scrollbar = tk.Scrollbar(display_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        h_scrollbar = tk.Scrollbar(display_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Frame for ASCII content
        self.ascii_frame = tk.Frame(self.canvas, bg="#000000")
        self.canvas.create_window((0, 0), window=self.ascii_frame, anchor=tk.NW)
        
        # ASCII label
        self.ascii_label = tk.Label(self.ascii_frame, text="", 
                                    bg="#000000", fg="#00ff00",
                                    font=("Courier", self.font_size),
                                    justify=tk.LEFT)
        self.ascii_label.pack()
        
        # Status bar
        self.status_bar = tk.Label(self.root, text="Ready", 
                                   bg="#007acc", fg="white",
                                   anchor=tk.W, padx=10)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def open_video(self):
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"), ("All files", "*.*")]
        )
        
        if file_path:
            self.video_path = file_path
            self.file_label.config(text=os.path.basename(file_path))
            self.btn_play.config(state=tk.NORMAL)
            self.status_bar.config(text=f"Loaded: {file_path}")
            
            # Initialize video capture
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(file_path)
            
    def frame_to_ascii(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate height to maintain aspect ratio
        aspect_ratio = 0.55  # Adjust for character aspect ratio
        height = int(gray.shape[0] * self.ascii_width / gray.shape[1] * aspect_ratio)
        
        # Resize
        resized = cv2.resize(gray, (self.ascii_width, height))
        
        # Convert to ASCII
        ascii_str = ""
        for row in resized:
            for pixel in row:
                # Map pixel value to character
                index = int(pixel / 255 * (len(self.chars) - 1))
                index = max(0, min(index, len(self.chars) - 1))
                ascii_str += self.chars[index]
            ascii_str += "\n"
        
        return ascii_str
    
    def update_frame(self):
        if self.playing and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            
            if not ret:
                # Loop video
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            
            if ret:
                # Convert frame to ASCII
                ascii_art = self.frame_to_ascii(frame)
                
                # Update label
                self.ascii_label.config(text=ascii_art)
                
                # Update scroll region
                self.ascii_frame.update_idletasks()
                self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
                
                # Update frame count
                frame_count = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.status_bar.config(text=f"Frame: {frame_count}/{total_frames}")
                
                # Schedule next update
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 24
                delay = int(1000 / fps)
                self.root.after(delay, self.update_frame)
    
    def toggle_playback(self):
        if not self.playing:
            self.playing = True
            self.btn_play.config(text="⏸ Pause", bg="#2d2d2d")
            self.btn_stop.config(state=tk.NORMAL)
            self.update_frame()
        else:
            self.playing = False
            self.btn_play.config(text="▶ Play", bg="#2d2d2d")
    
    def stop_playback(self):
        self.playing = False
        self.btn_play.config(text="▶ Play", state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        self.ascii_label.config(text="")
        self.status_bar.config(text="Stopped")
    
    def apply_settings(self):
        try:
            self.ascii_width = int(self.width_var.get())
            self.font_size = int(self.font_var.get())
            self.ascii_label.config(font=("Courier", self.font_size))
            self.status_bar.config(text="Settings applied")
        except ValueError:
            messagebox.showerror("Error", "Invalid settings values!")

if __name__ == "__main__":
    root = tk.Tk()
    app = ASCIIVideoGUI(root)
    root.mainloop()
