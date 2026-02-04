"""
GUI Application for Blood Cell Detection
Uses Tkinter for the interface
"""
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading

from detection import BloodCellDetector
from color_fusion import ColorSpaceFusion
from utils import load_image, save_image, convert_to_rgb


class BloodCellApp:
    """
    GUI Application for Blood Cell Detection and Analysis
    """
    
    def __init__(self):
        """Initialize the GUI application"""
        self.root = tk.Tk()
        self.root.title("Blood Cell Detection & Classification")
        self.root.geometry("1200x800")
        
        # Initialize detector and fusion
        self.detector = None
        self.fusion = None
        self.current_image = None
        self.result_image = None
        self.current_detections = []
        
        # Setup GUI
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup the user interface"""
        # Menu Bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Image", command=self.load_image)
        file_menu.add_command(label="Save Results", command=self.save_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Top Frame - Controls
        control_frame = tk.Frame(self.root, bg="#f0f0f0", height=80)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        control_frame.pack_propagate(False)
        
        # Buttons
        btn_frame = tk.Frame(control_frame, bg="#f0f0f0")
        btn_frame.pack(pady=10)
        
        self.load_btn = tk.Button(btn_frame, text="📁 Load Image", 
                                   command=self.load_image,
                                   bg="#4CAF50", fg="white", 
                                   font=("Arial", 12, "bold"),
                                   width=15, height=2)
        self.load_btn.pack(side=tk.LEFT, padx=5)
        
        self.analyze_btn = tk.Button(btn_frame, text="🔍 Analyze", 
                                      command=self.analyze_image,
                                      bg="#2196F3", fg="white",
                                      font=("Arial", 12, "bold"),
                                      width=15, height=2,
                                      state=tk.DISABLED)
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_btn = tk.Button(btn_frame, text="💾 Save Results", 
                                   command=self.save_results,
                                   bg="#FF9800", fg="white",
                                   font=("Arial", 12, "bold"),
                                   width=15, height=2,
                                   state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        # Main Content Frame
        content_frame = tk.Frame(self.root)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left Panel - Original Image
        left_panel = tk.LabelFrame(content_frame, text="Original Image", 
                                    font=("Arial", 12, "bold"))
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.original_canvas = tk.Canvas(left_panel, bg="white")
        self.original_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right Panel - Results
        right_panel = tk.LabelFrame(content_frame, text="Detection Results", 
                                     font=("Arial", 12, "bold"))
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        self.result_canvas = tk.Canvas(right_panel, bg="white")
        self.result_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Bottom Panel - Statistics
        stats_frame = tk.LabelFrame(self.root, text="Detection Statistics", 
                                     font=("Arial", 12, "bold"),
                                     height=150)
        stats_frame.pack(fill=tk.X, padx=10, pady=10)
        stats_frame.pack_propagate(False)
        
        # Statistics Labels
        stats_content = tk.Frame(stats_frame)
        stats_content.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Create statistics display
        stat_row1 = tk.Frame(stats_content)
        stat_row1.pack(fill=tk.X, pady=5)
        
        self.total_cells_label = self._create_stat_label(stat_row1, "Total Cells:", "0")
        self.rbc_label = self._create_stat_label(stat_row1, "RBCs:", "0", color="#FF0000")
        
        stat_row2 = tk.Frame(stats_content)
        stat_row2.pack(fill=tk.X, pady=5)
        
        self.wbc_label = self._create_stat_label(stat_row2, "WBCs:", "0", color="#0000FF")
        self.platelet_label = self._create_stat_label(stat_row2, "Platelets:", "0", color="#00FF00")
        
        # Status Bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = tk.Label(self.root, textvariable=self.status_var, 
                              bd=1, relief=tk.SUNKEN, anchor=tk.W,
                              font=("Arial", 10))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def _create_stat_label(self, parent, label_text, value_text, color="#000000"):
        """Create a statistics label"""
        frame = tk.Frame(parent)
        frame.pack(side=tk.LEFT, padx=20)
        
        label = tk.Label(frame, text=label_text, font=("Arial", 11, "bold"))
        label.pack(side=tk.LEFT, padx=5)
        
        value = tk.Label(frame, text=value_text, font=("Arial", 14, "bold"), 
                        fg=color)
        value.pack(side=tk.LEFT, padx=5)
        
        return value
        
    def load_image(self):
        """Load an image file"""
        file_path = filedialog.askopenfilename(
            title="Select Blood Smear Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), 
                      ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            self.status_var.set(f"Loading image: {Path(file_path).name}")
            self.current_image = load_image(file_path)
            
            # Display original image
            self._display_image(self.current_image, self.original_canvas)
            
            # Enable analyze button
            self.analyze_btn.config(state=tk.NORMAL)
            
            # Reset results
            self.result_canvas.delete("all")
            self._reset_statistics()
            
            self.status_var.set(f"Loaded: {Path(file_path).name}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
            self.status_var.set("Error loading image")
    
    def analyze_image(self):
        """Analyze the loaded image"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
        
        # Disable buttons during processing
        self.analyze_btn.config(state=tk.DISABLED)
        self.load_btn.config(state=tk.DISABLED)
        
        # Run analysis in separate thread
        threading.Thread(target=self._run_analysis, daemon=True).start()
    
    def _run_analysis(self):
        """Run the analysis in background thread"""
        try:
            print("\n" + "="*60)
            print("STARTING ANALYSIS")
            print("="*60)
            
            self.status_var.set("Initializing models...")
            
            # Initialize models if not already done
            if self.detector is None:
                print("Initializing detector and fusion models...")
                self.detector = BloodCellDetector()
                self.fusion = ColorSpaceFusion()
                print("✓ Models initialized")
            
            self.status_var.set("Detecting cells...")
            print("\nRunning YOLO detection...")
            
            # Run detection
            detections, crops = self.detector.detect(self.current_image, return_crops=True)
            print(f"✓ YOLO detected {len(detections)} objects")
            
            self.status_var.set(f"Detected {len(detections)} cells. Refining classification...")
            
            # Refine with color fusion
            print("\nRefining classifications with color fusion...")
            refined_detections = []
            for i, (det, crop) in enumerate(zip(detections, crops)):
                if crop is not None and crop.size > 0:
                    print(f"  Processing cell {i+1}/{len(detections)}...")
                    refined_class, confidence = self.fusion.classify_cell(crop)
                    det['refined_class'] = refined_class
                    det['refined_confidence'] = confidence
                    refined_detections.append(det)
            
            print(f"✓ Refined {len(refined_detections)} detections")
            
            self.current_detections = refined_detections
            
            # Draw results
            print("\nDrawing results...")
            self.result_image = self._draw_detections(self.current_image.copy(), 
                                                      refined_detections)
            print("✓ Results ready")
            
            # Update UI on main thread
            self.root.after(0, self._update_results)
            
            print("\n" + "="*60)
            print("ANALYSIS COMPLETE")
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"\n{'!'*60}")
            print("ERROR DURING ANALYSIS")
            print(f"{'!'*60}")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            print(f"{'!'*60}\n")
            
            error_msg = f"Analysis failed:\n\n{str(e)}\n\nCheck the console for details."
            self.root.after(0, lambda: messagebox.showerror("Analysis Error", error_msg))
            self.root.after(0, lambda: self.status_var.set(f"Error: {str(e)}"))
        finally:
            # Re-enable buttons
            self.root.after(0, lambda: self.analyze_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.load_btn.config(state=tk.NORMAL))
    
    def _update_results(self):
        """Update the UI with analysis results"""
        # Display result image
        self._display_image(self.result_image, self.result_canvas)
        
        # Update statistics
        counts = {'RBC': 0, 'WBC': 0, 'Platelet': 0}
        for det in self.current_detections:
            cell_class = det.get('refined_class', det.get('class', 'Unknown'))
            counts[cell_class] = counts.get(cell_class, 0) + 1
        
        self.total_cells_label.config(text=str(len(self.current_detections)))
        self.rbc_label.config(text=str(counts.get('RBC', 0)))
        self.wbc_label.config(text=str(counts.get('WBC', 0)))
        self.platelet_label.config(text=str(counts.get('Platelet', 0)))
        
        # Enable save button
        self.save_btn.config(state=tk.NORMAL)
        
        self.status_var.set(f"Analysis complete! Detected {len(self.current_detections)} cells")
    
    def _draw_detections(self, image, detections):
        """Draw bounding boxes on image"""
        colors = {
            'RBC': (0, 0, 255),
            'WBC': (255, 0, 0),
            'Platelet': (0, 255, 0),
            'Unknown': (128, 128, 128)
        }
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            class_name = det.get('refined_class', det.get('class', 'Unknown'))
            confidence = det.get('refined_confidence', det.get('confidence', 0.0))
            
            color = colors.get(class_name, (255, 255, 255))
            
            # Draw box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return image
    
    def _display_image(self, image, canvas):
        """Display image on canvas"""
        # Convert BGR to RGB
        if len(image.shape) == 3:
            display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            display_image = image
        
        # Resize to fit canvas
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width < 10:  # Canvas not rendered yet
            canvas_width = 500
            canvas_height = 400
        
        # Calculate scaling
        h, w = display_image.shape[:2]
        scale = min(canvas_width / w, canvas_height / h) * 0.95
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize image
        resized = cv2.resize(display_image, (new_w, new_h))
        
        # Convert to PhotoImage
        pil_image = Image.fromarray(resized)
        photo = ImageTk.PhotoImage(pil_image)
        
        # Display on canvas
        canvas.delete("all")
        canvas.create_image(canvas_width // 2, canvas_height // 2, 
                          image=photo, anchor=tk.CENTER)
        canvas.image = photo  # Keep reference
    
    def save_results(self):
        """Save the result image"""
        if self.result_image is None:
            messagebox.showwarning("Warning", "No results to save!")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                save_image(self.result_image, file_path)
                messagebox.showinfo("Success", f"Results saved to:\n{file_path}")
                self.status_var.set(f"Saved: {Path(file_path).name}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save:\n{str(e)}")
    
    def _reset_statistics(self):
        """Reset all statistics to 0"""
        self.total_cells_label.config(text="0")
        self.rbc_label.config(text="0")
        self.wbc_label.config(text="0")
        self.platelet_label.config(text="0")
        self.save_btn.config(state=tk.DISABLED)
    
    def run(self):
        """Run the application"""
        self.root.mainloop()


if __name__ == "__main__":
    app = BloodCellApp()
    app.run()