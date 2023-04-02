import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import Tools
import Classification
import Tools2


class ImageEditor(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Reconocimiento de Patrones: Clasificación Bayesiana")
        self.master.geometry("1200x800")
        self.master.configure()
        self.pack()

        self.open_button = tk.Button(self, text="Open", command=self.open_images)
        self.label_textbox_classes = tk.Label(self, text="Select the number of classes on the current classification: ")
        self.textbox_classes = tk.Entry(self)

        self.open_button.pack(side="top")
        self.label_textbox_classes.pack_forget()
        self.textbox_classes.pack_forget()

        self.images = []
        self.canvases = []
        self.num_classes = None
        self.filenames = []

    def open_images(self):
        self.filenames = filedialog.askopenfilenames()
        if self.filenames:
            self.label_textbox_classes.pack(side="top")
            self.textbox_classes.pack(side="top")

            self.textbox_classes.focus_set()
            self.textbox_classes.bind("<Return>", self.process_image)
            

    def process_image(self, event):
        if self.textbox_classes.get() != "":
            self.num_classes = int(self.textbox_classes.get())
        i = 0
        x = 0
        for filename in self.filenames:
            # for x in range(self.num_classes):
            image = Tools.GaussianFilter(filename)
        
            # Tools.CropClasses(filename.rstrip(".jpg")+".png", x+1, i+1)
            Tools2.Mask_Colors(filename.rstrip(".jpg")+".png", i)

            # output_image = f"Class_{i+1}_Contour_{j+1}.png"
            # output_image = Image.open(output_image)
            # newsize = (400,400)
            # output_image = output_image.resize(newsize)

            # output_mask = f"output_mask_class_{x+1}_{i+1}.png"
            # output_mask = Image.open(output_mask)

            # self.images.append(output_image)
            # photo = ImageTk.PhotoImage(output_image)
            # self.images.append(output_mask)
            # photo2 = ImageTk.PhotoImage(output_mask)
            # canvas = tk.Canvas(self, width=1200, height=1200)
            # canvas.create_image(100,100, image=photo, anchor="nw")
            # canvas.create_image(601, 100, image=photo2, anchor="nw")
            # canvas.pack()
            # self.canvases.append(canvas)

            # canvas.bind("<Button-3>", lambda event: canvas.destroy())
            # canvas.wait_window()

            i += 1



        # Classification.BayesRGB(self.images, self.num_classes) 



root = tk.Tk()
app = ImageEditor(master=root)
app.mainloop()
