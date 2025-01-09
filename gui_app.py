import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
from pathlib import Path
from src.inference import predict_image
import logging

class PizzaDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pizza Detector")
        self.root.geometry("800x600")
        
        self.main_frame = tk.Frame(self.root, padx=20, pady=20)
        self.main_frame.pack(expand=True, fill='both')
        
        title = tk.Label(self.main_frame, text="Pizza Detector", font=('Arial', 24, 'bold'))
        title.pack(pady=20)
        
        self.image_frame = tk.Frame(self.main_frame)
        self.image_frame.pack(pady=20)
        
        self.image_label = tk.Label(self.image_frame, text="No image selected", 
                                  width=400, height=300, relief='solid')
        self.image_label.pack()
        
        button_frame = tk.Frame(self.main_frame)
        button_frame.pack(pady=20)
        
        self.load_button = tk.Button(button_frame, text="Load Image", 
                                   command=self.load_image, width=15)
        self.load_button.pack(side=tk.LEFT, padx=10)
        
        self.detect_button = tk.Button(button_frame, text="Detect Pizza", 
                                     command=self.detect_pizza, width=15)
        self.detect_button.pack(side=tk.LEFT, padx=10)
        
        self.result_label = tk.Label(self.main_frame, text="", font=('Arial', 12))
        self.result_label.pack(pady=20)
        
        self.current_image_path = None
        self.photo = None

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                image = Image.open(file_path)
                image = image.resize((400, 300), Image.Resampling.LANCZOS)
                self.photo = ImageTk.PhotoImage(image)
                
                self.image_label.config(image=self.photo)
                self.image_label.image = self.photo
                
                self.current_image_path = file_path
                self.result_label.config(text="")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error loading image: {str(e)}")

    def detect_pizza(self):
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
            
        try:
            confidence, label = predict_image(self.current_image_path)
            result_text = f"Prediction: {label}\nConfidence: {confidence:.2f}%"
            self.result_label.config(text=result_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error during prediction: {str(e)}")

def main():
    root = tk.Tk()
    app = PizzaDetectorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()