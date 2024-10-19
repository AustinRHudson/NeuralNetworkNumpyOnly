import tkinter as tk
from tkinter import ALL, EventType
from PIL import Image, ImageGrab

def paint(e):
    x1, y1 = e.x-1, e.y-1
    x2, y2 = e.x, e.y
    canvas.create_line(x1,y1,x2,y2,fill='white', width=3, capstyle="round", arrow=tk.BOTH)

def do_zoom(event):
    x = canvas.canvasx(event.x)
    y = canvas.canvasy(event.y)
    factor = 1.001 ** event.delta
    canvas.scale(ALL, x, y, factor, factor)

def getter(widget):
    x=root.winfo_rootx()+widget.winfo_x()
    y=root.winfo_rooty()+widget.winfo_y()
    x1=x+widget.winfo_width()
    y1=y+widget.winfo_height()
    ImageGrab.grab().crop((x,y,x1,y1)).save("testImage.png")

root = tk.Tk()
root.title("Digit Recognition Demo")
root.geometry("800x800")

brushColor = "black"

canvas = tk.Canvas(root, width=800, height=800, bg="black")
canvas.create_bitmap((28,28))
canvas.bind('<B1-Motion>', paint)
canvas.bind('<Button-2>', getter(root))
canvas.bind("<MouseWheel>", do_zoom) # WINDOWS ONLY
canvas.pack()

root.mainloop()