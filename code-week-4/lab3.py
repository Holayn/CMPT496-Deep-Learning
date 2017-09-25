#Importing
import numpy as np
from scipy import signal
from scipy import misc
import matplotlib.pyplot as plt
from PIL import Image

### Load image fox.jpg

im = Image.open('fox.jpg')  # type here your image's name

# uses the ITU-R 601-2 Luma transform (there are several
# ways to convert an image to grey scale)

image_gr = im.convert("L")
print("\n Original type: %r \n\n" % image_gr)
# convert image to a matrix with values from 0 to 255 (uint8)
arr = np.asarray(image_gr)
print("After conversion to numerical representation: \n\n %r" % arr)

### Plot image

imgplot = plt.imshow(arr)
imgplot.set_cmap('gray')  #you can experiment different colormaps (Greys,winter,autumn)
print("\n Input image converted to gray scale: \n")
plt.show(imgplot)

kernel = np.array([[ 0, 1, 0],[ 1,-4, 1],[ 0, 1, 0]])
#This kernel is first derivative filter
#Same as [-1,1], but highlights more of the edges
#Negative surrounded by positive, vice verse
#Works for horizontal and vertical

grad = signal.convolve2d(arr, kernel, mode='same', boundary='symm')
print('GRADIENT MAGNITUDE - Feature map')

fig, aux = plt.subplots(figsize=(10, 10))
aux.imshow(np.absolute(grad), cmap='gray')
#edge highlighting
plt.show(aux)


type(grad)

grad_biases = np.absolute(grad) + 100

grad_biases[grad_biases > 255] = 255


print('GRADIENT MAGNITUDE - Feature map')

fig, aux = plt.subplots(figsize=(10, 10))
aux.imshow(np.absolute(grad_biases), cmap='gray')
plt.show(aux)
#edges are now more evident
