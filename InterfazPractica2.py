import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class ImageEditor(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Reconocimiento de Patrones: Clasificación Bayesiana")
        self.master.configure(width=300, height=300)
        self.master.configure(bg="black")
        self.pack()
        
        # Create widgets
        self.open_button = tk.Button(self, text="Abrir Imágenes", command=self.open_images)
        self.save_button = tk.Button(self, text="Guardar", command=self.save_images)
        self.canvas = tk.Canvas(self, width=800, height=800, bg="black")

        # Add widgets to layout
        self.open_button.pack(side="top")
        self.save_button.pack(side="top")
        self.canvas.pack()

        # Initialize variables
        self.images = []
        self.canvases = []

    def open_images(self):
        # Let the user choose files to open
        filenames = filedialog.askopenfilenames()
        if filenames:
            for filename in filenames:
                image = Image.open(filename)
                image = image.resize((400, 400), Image.ANTIALIAS)
                photo_image = ImageTk.PhotoImage(image)
                self.images.append(image)
                self.photo_images.append(photo_image)
                self.canvas.create_image(0, 0, image=photo_image, anchor="nw")

    def save_images(self):
        if self.images:
            # Let the user choose a folder to save the images
            foldername = filedialog.askdirectory()
            if foldername:
                for i, image in enumerate(self.images):
                    # Save each image to a file in the chosen folder
                    save_filename = f"{foldername}/image{i}.png"
                    image.save(save_filename)

root = tk.Tk()
app = ImageEditor(master=root)
app.mainloop()
