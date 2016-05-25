

from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import imreg_dft as ird

im0 = imresize(imread(standard_image, flatten=True),.25)

im1 = imresize(imread(image_to_be_registered, flatten=True),.25)

result = ird.similarity(im0, im1, numiter=3)

plt.subplot(211)
plt.imshow(im1, cmap='gray')

plt.subplot(212)
plt.imshow(result['timg'], cmap='gray')

plt.show()
