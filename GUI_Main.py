import tkinter as tk
from tkinter import ttk
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
        self.width = 1543 
        self.height = 679
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        self.x = (self.screen_width / 2) - (self.width / 2)
        self.y = (self.screen_height / 2) - (self.height / 2)
        self.master.geometry("%dx%d+%d+%d" % (self.width, self.height, self.x, self.y))
        #self.master.geometry("1200x800")
        self.master.configure()
        self.pack()

        self.bg_image = tk.PhotoImage(file="Extras/background3.png")
        self.bg_label = tk.Label(root, image=self.bg_image)
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)


        self.program_var = tk.StringVar()
        self.program_combobox = ttk.Combobox(root, textvariable=self.program_var, font=("MS Sans Serif", 12))
        self.program_combobox['values'] = ("monkey_classification", "fruit_classification")
        self.program_combobox.pack(side="right",padx=100,pady=280)

        # self.open_button = tk.Button(root, text="Open Files", font=("MS Sans Serif", 12), command=self.process_fruit_image)
        self.program_var.trace_add("write", lambda *args: self.update_button())

        self.images = []
        self.canvases = []
        self.num_classes = None
        self.filenames = []
    

    # function to update the button visibility based on the selected program
    def update_button(self):
        if self.program_var.get() == "monkey_classification" :
            # self.open_button.pack(side='right',pady=20)
            self.process_monkey_image()

        elif self.program_var.get() == "fruit_classification":
            # self.open_button.pack(side='right',pady=20)
            self.process_fruit_image()

        else:
            self.open_button.pack_forget()


    def process_fruit_image(self):
        self.filenames = filedialog.askopenfilenames()
        i = 0
        x = 0
        for filename in self.filenames:
            image = Tools.GaussianFilter(filename)
        
            Tools2.Fruit_Mask(filename.rstrip(".jpg")+".png", i)

            i += 1

        Classification.BayesRGB(2)

    def process_monkey_image(self):
        self.filenames = filedialog.askopenfilenames()
        i = 0
        x = 0
        for filename in self.filenames:
            image = Tools.GaussianFilter(filename)
            Tools2.Monkey_Mask(filename.rstrip(".jpg")+".png", i)

            i += 1
        Classification.BayesRGB(1)


root = tk.Tk()
app = ImageEditor(master=root)
app.mainloop()
