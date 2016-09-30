"""
Daniel Buscombe, June - September 2016
"""

from __future__ import division, print_function
import os, time
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize, toimage
import math

try:
    import scipy.ndimage.interpolation as ndii
except ImportError:
    import ndimage.interpolation as ndii

import numpy as np
from PIL import Image

from fnmatch import filter 

from joblib import Parallel, delayed, cpu_count

import warnings
warnings.filterwarnings("ignore")

#from Tkinter import Tk
#from tkFileDialog import askopenfilename, askdirectory


#==============================================
#==============================================
def translation(im0, im1):
    """Return translation vector to register images."""
    shape = im0.shape
    f0 = np.fft.fft2(im0)
    f1 = np.fft.fft2(im1)
    ir = abs(np.fft.ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))
    t0, t1 = np.unravel_index(np.argmax(ir), shape)
    if t0 > shape[0] // 2:
        t0 -= shape[0]
    if t1 > shape[1] // 2:
        t1 -= shape[1]
    return [t0, t1]

#==============================================
def highpass(shape):
    """Return highpass filter to be multiplied with fourier transform."""
    x = np.outer(
        np.cos(np.linspace(-math.pi/2., math.pi/2., shape[0])),
        np.cos(np.linspace(-math.pi/2., math.pi/2., shape[1])))
    return (1.0 - x) * (2.0 - x)

#==============================================
def logpolar(image, angles=None, radii=None):
    """Return log-polar transformed image and log base."""
    shape = image.shape
    center = shape[0] / 2, shape[1] / 2
    if angles is None:
        angles = shape[0]
    if radii is None:
        radii = shape[1]
    theta = np.empty((angles, radii), dtype=np.float64)
    theta.T[:] = -np.linspace(0, np.pi, angles, endpoint=False)
    #d = radii
    d = np.hypot(shape[0]-center[0], shape[1]-center[1])
    log_base = 10.0 ** (math.log10(d) / (radii))
    radius = np.empty_like(theta)
    radius[:] = np.power(log_base, np.arange(radii, dtype=np.float64)) - 1.0
    x = radius * np.sin(theta) + center[0]
    y = radius * np.cos(theta) + center[1]
    output = np.empty_like(x)
    ndii.map_coordinates(image, [x, y], output=output)
    return output, log_base

#==============================================
def similarity(im0, im1):
    """Return similarity transformed image im1 and transformation parameters.

    Transformation parameters are: isotropic scale factor, rotation angle (in
    degrees), and translation vector.

    A similarity transformation is an affine transformation with isotropic
    scale and without shear.

    Limitations:
    Image shapes must be equal and square.
    All image areas must have same scale, rotation, and shift.
    Scale change must be less than 1.8.
    No subpixel precision.

    """
    if im0.shape != im1.shape:
        raise ValueError("Images must have same shapes.")
    elif len(im0.shape) != 2:
        raise ValueError("Images must be 2 dimensional.")

    f0 = np.fft.fftshift(abs(np.fft.fft2(im0)))
    f1 = np.fft.fftshift(abs(np.fft.fft2(im1)))

    h = highpass(f0.shape)
    f0 *= h
    f1 *= h
    del h

    f0, log_base = logpolar(f0)
    f1, log_base = logpolar(f1)

    f0 = np.fft.fft2(f0)
    f1 = np.fft.fft2(f1)
    r0 = abs(f0) * abs(f1)
    ir = abs(np.fft.ifft2((f0 * f1.conjugate()) / r0))
    i0, i1 = np.unravel_index(np.argmax(ir), ir.shape)
    angle = 180.0 * i0 / ir.shape[0]
    scale = log_base ** i1

    if scale > 1.8:
        ir = abs(np.fft.ifft2((f1 * f0.conjugate()) / r0))
        i0, i1 = np.unravel_index(np.argmax(ir), ir.shape)
        angle = -180.0 * i0 / ir.shape[0]
        scale = 1.0 / (log_base ** i1)
        if scale > 1.8:
            raise ValueError("Images are not compatible. Scale change > 1.8")

    if angle < -90.0:
        angle += 180.0
    elif angle > 90.0:
        angle -= 180.0

    im2 = ndii.zoom(im1, 1.0/scale)
    im2 = ndii.rotate(im2, angle)

    if im2.shape < im0.shape:
        t = np.zeros_like(im0)
        t[:im2.shape[0], :im2.shape[1]] = im2
        im2 = t
    elif im2.shape > im0.shape:
        im2 = im2[:im0.shape[0], :im0.shape[1]]

    f0 = np.fft.fft2(im0)
    f1 = np.fft.fft2(im2)
    ir = abs(np.fft.ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))
    t0, t1 = np.unravel_index(np.argmax(ir), ir.shape)

    if t0 > f0.shape[0] // 2:
        t0 -= f0.shape[0]
    if t1 > f0.shape[1] // 2:
        t1 -= f0.shape[1]

    im2 = ndii.shift(im2, [t0, t1])

    # correct parameters for ndimage's internal processing
    if angle > 0.0:
        d = int((int(im1.shape[1] / scale) * math.sin(math.radians(angle))))
        t0, t1 = t1, d+t0
    elif angle < 0.0:
        d = int((int(im1.shape[0] / scale) * math.sin(math.radians(angle))))
        t0, t1 = d+t1, d+t0
    scale = (im1.shape[1] - 1) / (int(im1.shape[1] / scale) - 1)

    return im2, scale, angle, [t0, t1]


#==============================================
def doproc(master, direc, outdirec, slave):

    outfile = outdirec + slave.split(os.sep)[-1].split(os.path.splitext(slave)[1])[0]+'_reg.jpg'

    if not os.path.isfile(outfile):
       rgbimage = np.asarray(Image.open(direc+slave))
       slave_image = np.asarray(Image.open(direc+slave).convert('L'))

       try:
          t0, t1 = translation(master, slave_image)
          tvec = [t0, t1]

          if np.sum(np.abs(tvec))>16: 

             imtemp = ndii.shift(rgbimage, [t0, t1,0])

             #imtemp, scale, angle, (t1,t0) = similarity(master, slave_image)
             #tvec = [t0, t1]

             #imtemp = ndii.zoom(rgbimage, 1.0/scale)
             #imtemp = ndii.rotate(rgbimage, angle)

             #if imtemp.shape < rgbimage.shape:
             #   t = np.zeros_like(rgbimage)
             #   t[:imtemp.shape[0], :imtemp.shape[1]] = imtemp
             #   imtemp = t
             #elif imtemp.shape > rgbimage.shape:
             #   imtemp = imtemp[:rgbimage.shape[0], :rgbimage.shape[1]]

             #imtemp = ndii.shift(imtemp, [t0, t1, 0])

             toimage(imtemp).save(outfile)

             with open(outfile.split('.jpg')[0]+'_xy.txt', 'w') as f:
                np.savetxt(f, [t0, t1], delimiter=',', fmt="%8.6f")

          else:
             toimage(rgbimage).save(outfile)

       except:
          toimage(rgbimage).save(outfile)

#==============================================
#==============================================
if __name__ == '__main__':

   master_file = '/run/media/dbuscombe/MASTER/GCMRC/SANDBAR_REMOTECAMERAS/RC1194R/deployment8/RC1194R_20151003_1151.JPG'

   direc = '/run/media/dbuscombe/MASTER/GCMRC/SANDBAR_REMOTECAMERAS/RC1194R/deployment8/'
   filenames = sorted(filter(os.listdir(direc), '*.[Jj][Pp][Gg]'))

   #Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
   #filenames = askopenfilename(filetypes=[("images","*.jpg *.JPG")], multiple=True)

   outdirec = '/run/media/dbuscombe/MASTER/GCMRC/SANDBAR_REMOTECAMERAS/RC1194R_regis/deployment8/'

   try:
      os.mkdir(outdirec)
   except:
      pass

   master = np.asarray(Image.open(master_file).convert('L'))

   Parallel(n_jobs = cpu_count()-1, verbose=1000)(delayed(doproc)(master, direc, outdirec, slave) for slave in filenames)




