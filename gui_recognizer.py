import tkinter as tk
from tkinter import *
from PIL import ImageGrab, Image, ImageOps
import numpy as np
from keras.models import load_model

model = load_model('mnist.h5')

def predict_digit(img):
	# the image needs to be resized to its 28x28 pixels form
	img = img.resize((28, 28))
	# since the image may be coloured it needs to go from rgb to grayscale
	img = img.convert('L')
	img = np.array(img)
	# since our model input has a certain shape to it - (1, 28, 28, 1) - we need to reshape and normalize
	img = img.reshape(1, 28, 28, 1)
	img = img / 255.0
	res = model.predict([img])[0]
	# np.argmax(res) is the predicted digit
	# max(res) is the accuracy of the predicted digit
	return np.argmax(res), 1 - max(res)

class App(tk.Tk):
	def __init__(self):
		tk.Tk.__init__(self)
		self.x = self.y = 0
		# this is the creation of all the elements that will be seen and be interacted with
		self.canvas = tk.Canvas(self, width = 300, height = 300, bg = "black", cursor = "cross")
		self.label = tk.Label(self, text = "Thinking...", font = ("Arial", 50))
		self.classify_btn = tk.Button(self, text = "Recognize", command = self.classify_handwriting)
		self.button_clear = tk.Button(self, text = "Clear", command = self.clear_all)
		
		# grid structure
		self.canvas.grid(row = 0, column = 0, pady = 2, sticky = W)
		self.label.grid(row = 0, column = 1, pady = 2, padx = 2)
		self.classify_btn.grid(row = 1, column = 1, pady = 2, padx = 2)
		self.button_clear.grid(row = 1, column = 0, pady = 2)

		self.canvas.bind("<B1-Motion>", self.draw_lines)
	
	def clear_all(self):
		self.canvas.delete("all")

	def classify_handwriting(self):
		HWND = self.canvas.winfo_id()
		x, y = (self.canvas.winfo_rootx(), self.canvas.winfo_rooty())
		width, height = (self.canvas.winfo_width(), self.canvas.winfo_height())
		rect = (x, y, x + width, y + height)
		im = ImageGrab.grab(rect)
		
		digit, acc = predict_digit(im)
		self.label.configure(text = str(digit) + ', ' + str(int(acc * 100)) + '%')
	
	def draw_lines(self, event):
		self.x = event.x
		self.y = event.y
		r = 8
		self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill = 'white')
app = App()
mainloop()

