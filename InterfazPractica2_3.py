
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class ImageEditor(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()

        # Create widgets
        self.open_button = tk.Button(self, text="Open", command=self.open_images)
        self.save_button = tk.Button(self, text="Save", command=self.save_images)

        # Add widgets to layout
        self.open_button.pack(side="top")
        self.save_button.pack(side="top")

        # Initialize variables
        self.images = []
        self.canvases = []
        self.menu_items = []

        # Create a frame for the image menu
        self.menu_frame = tk.Frame(self)
        self.menu_frame.pack(side="top")

    def update_menu(self):
        # Remove all existing menu items
        for item in self.menu_items:
            item.destroy()
        self.menu_items.clear()

        # Add menu items for each image
        for i, image in enumerate(self.images):
            # Create a miniature of the image
            thumbnail = image.copy().resize((100, 100))
            photo = ImageTk.PhotoImage(thumbnail)
            label = tk.Label(self.menu_frame, image=photo)

            # Bind a click event to select the image
            label.bind("<Button-1>", lambda event, index=i: self.select_image(index))

            # Add the label to the menu frame
            label.pack(side="left")

            # Store the photo object in the list of menu items
            self.menu_items.append(label)
            self.menu_items.append(photo)

    def open_images(self):
        # Let the user choose files to open
        filenames = filedialog.askopenfilenames()
        if filenames:
            for filename in filenames:
                # Load the image and create a new canvas to display it
                image = Image.open(filename)
                self.images.append(image)
                self.update_menu()

    def select_image(self, index):
        # Clear all existing canvases
        for canvas in self.canvases:
            canvas.destroy()
        self.canvases = []

        # Create a new canvas to display the selected image
        image = self.images[index]
        photo = ImageTk.PhotoImage(image)
        canvas = tk.Canvas(self, width=photo.width(), height=photo.height())
        canvas.create_image(0, 0, image=photo, anchor="nw")
        canvas.pack()
        self.canvases.append(canvas)
