import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import Tools
import Classification
import Tools2




# create the main window
root = tk.Tk()
root.title("Bayesian Image Classification")

# set the window size and position
width = 1543 
height = 679
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width / 2) - (width / 2)
y = (screen_height / 2) - (height / 2)
root.geometry("%dx%d+%d+%d" % (width, height, x, y))

# add a background image
bg_image = tk.PhotoImage(file="background3.png")
bg_label = tk.Label(root, image=bg_image)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)


program_var = tk.StringVar()
program_combobox = ttk.Combobox(root, textvariable=program_var, font=("MS Sans Serif", 12))
program_combobox['values'] = ("monkey_classification", "fruit_classification")
program_combobox.pack(side="right",padx=100,pady=280)

# function to open files
def open_files():
    files = filedialog.askopenfilenames()
    print(files)

# add an "open" button
open_button = tk.Button(root, text="Open Files", font=("MS Sans Serif", 12), command=open_files)

# function to update the button visibility based on the selected program
def update_button():
    if program_var.get() == "monkey_classification" :
        open_button.pack(side='right',pady=20)

    elif program_var.get() == "fruit_classification":
	 	open_button.pack(side='right',pady=20)

    else:
        open_button.pack_forget()

# bind the function to the combobox selection event
program_var.trace_add("write", lambda *args: update_button())

# run the main loop
root.mainloop()
