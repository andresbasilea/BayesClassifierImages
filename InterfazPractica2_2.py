import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import Tools


class ImageEditor(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Reconocimiento de Patrones: Clasificación Bayesiana")
        self.pack()

        # Create widgets
        self.open_button = tk.Button(self, text="Open", command=self.open_images)

        # Add widgets to layout
        self.open_button.pack(side="top")

        # Initialize variables
        self.images = []
        self.canvases = []

    def open_images(self):
        filenames = filedialog.askopenfilenames()
        if filenames:
            for filename in filenames:
                image = Tools.GaussianFilter(filename)
                self.images.append(image)
                photo = ImageTk.PhotoImage(image)
                canvas = tk.Canvas(self, width=photo.width(), height=photo.height())
                canvas.create_image(0, 0, image=photo, anchor="nw")
                canvas.pack()
                self.canvases.append(canvas)

                Tools.CropClasses(filename.rstrip(".jpg")+".png")


                # Wait for the user to close the canvas before opening the next image
                canvas.bind("<Button-3>", lambda event: canvas.destroy())
                canvas.wait_window()



root = tk.Tk()
app = ImageEditor(master=root)
app.mainloop()
