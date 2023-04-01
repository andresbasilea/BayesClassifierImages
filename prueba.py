import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import Tools


class ImageEditor(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Reconocimiento de Patrones: Clasificación Bayesiana")
        self.master.geometry("600x600")
        self.pack()

        # Create widgets
        self.open_button = tk.Button(self, text="Open", command=self.open_images)
        self.label_textbox_classes = tk.Label(self, text="Select the number of classes on the current image: ")
        self.textbox_classes = tk.Entry(self)


        # Add widgets to layout
        self.open_button.pack(side="top")
        self.label_textbox_classes.pack_forget()  # Hide initially
        self.textbox_classes.pack_forget()  # Hide initially

        # Initialize variables
        self.images = []
        self.canvases = []
        self.num_classes = None  # Initialize to None

    def open_images(self):
        filenames = filedialog.askopenfilenames()
        if filenames:
            self.label_textbox_classes.pack(side="top")
            self.textbox_classes.pack(side="top")

            # Wait for the user to enter the number of classes before processing images
            self.textbox_classes.focus_set()
            self.textbox_classes.bind("<Return>", self.process_images)

    def process_images(self, event):
        self.num_classes = int(self.textbox_classes.get())
        self.label_textbox_classes.pack_forget()
        self.textbox_classes.pack_forget()
        self.textbox_classes.delete(0, "end")  # Clear the textbox

        # Process each selected image
        for filename in filedialog.askopenfilenames():
            image = Tools.GaussianFilter(filename)
            self.images.append(image)
            photo = ImageTk.PhotoImage(image)

            canvas = tk.Canvas(self, width=photo.width(), height=photo.height())
            canvas.create_image(0, 0, image=photo, anchor="nw")
            canvas.pack()
            self.canvases.append(canvas)

            # Wait for the user to close the canvas before opening the next image
            canvas.bind("<Button-3>", lambda event: canvas.destroy())
            canvas.wait_window()

            Tools.CropClasses(filename.rstrip(".jpg")+".png")

        # Do something with self.num_classes and self.images here
        print("Number of classes:", self.num_classes)
        print("Images processed:", len(self.images))


root = tk.Tk()
app = ImageEditor(master=root)
app.mainloop()
