"""
Daniel Buscombe, June 2016
"""

from scipy.misc import imresize, toimage
import imreg_dft as ird
import os, time
import numpy as np
from scipy.linalg import inv
from PIL import Image

from sift import read_features_from_file, match, process_image
from ransac import get_points #, doransac

from fnmatch import filter 

from joblib import Parallel, delayed, cpu_count

import warnings
warnings.filterwarnings("ignore")


#==============================================
#for slave in filenames:
def doproc(im2, key2, direc, outdirec, w, h, slave):

    outfile = outdirec + slave.split(os.sep)[-1].split(os.path.splitext(slave)[1])[0]+'_reg.jpg'

    if not os.path.isfile(outfile): 

       im1 = Image.open(direc+slave).convert('L') #sys.argv[1]).convert('L')
        
       im1 = im1.resize((w/4, h/4))    
       im1.save('temp/1.pgm')
       im1 = np.asarray(im1)

       process_image('temp/1.pgm', 'temp/1.key')
       key1 = read_features_from_file('temp/1.key')
       score = match(key1[1], key2[1])
       plist = get_points(key1[0], key2[0], score)

       a,b = zip(*plist)
       shift = np.asarray(b)-np.asarray(a)
       tvec = np.mean(shift, axis=0)*4

#       T = []
#       for k in xrange(20):
#          try:
#             out = doransac(im1, im2, plist)
#             H = inv(out)

#             tvec=[-H[0][2]*4, -H[1][2]*4]
#             T.append(tvec)
#          except:
#             pass

       try:
          #tvec = np.percentile(T,50,axis=0)

          im1 = Image.open(direc+slave)

          if np.sum(np.abs(tvec))>16:      
             print "x dislocation: "+str(tvec[0])
             print "y dislocation: "+str(tvec[1])

             imtemp = ird.transform_img(np.asarray(im1), tvec= tvec) #[-H[0][2]*4, -H[1][2]*4])
             toimage(imtemp).save(outfile)

          else:
             toimage(np.asarray(im1)).save(outfile)
       except:
          pass


#==============================================
if __name__ == '__main__':

   master = '/run/media/dbuscombe/MASTER/GCMRC/SANDBAR_REMOTECAMERAS/RC0307Rf/RC0307Rf_20091012_1130.jpg'

   direc = '/run/media/dbuscombe/MASTER/GCMRC/SANDBAR_REMOTECAMERAS/RC0307Rf/'
   filenames = sorted(filter(os.listdir(direc), '*.[Jj][Pp][Gg]'))

   outdirec = '/run/media/dbuscombe/MASTER/GCMRC/SANDBAR_REMOTECAMERAS/RC0307Rf_regis/'

   try:
      os.mkdir("temp")
   except OSError:
      pass


   im2 = Image.open(master).convert('L') #sys.argv[2]).convert('L')
   w,h = im2.size

   im2 = im2.resize((w/4, h/4))    
   im2.save('temp/2.pgm')
   im2 = np.asarray(im2)
   process_image('temp/2.pgm', 'temp/2.key')
   key2 = read_features_from_file('temp/2.key')

   #the location of each keypoint in the image is specified by 4 floating point numbers giving subpixel row and column location,
   #scale, and orientation (in radians from -PI to PI).  
   #Obviously, these numbers are not invariant to viewpoint, but can be used in later
   #stages of processing to check for geometric consistency among matches.
   #Finally, the invariant descriptor vector for the keypoint is given as
   #a list of 128 integers in range [0,255].

   Parallel(n_jobs = cpu_count(), verbose=0)(delayed(doproc)(im2, key2, direc, outdirec, w, h, slave) for slave in filenames)



    ##22mile
    #master = '/home/filfy/github_clones/sandbar_seg/RC0220Ra_surveydata/RC0220Ra_20130922_1353.jpg'
    #slave =  '/home/filfy/github_clones/sandbar_seg/RC0220Ra_surveydata/RC0220Ra_20150924_1141.JPG'

#    master = '/home/filfy/github_clones/sandbar_seg/RC0089L_surveydata/RC0089L_20121004_0959.jpg'
#    slave = '/home/filfy/github_clones/sandbar_seg/RC0089L_surveydata/RC0089L_20130921_1553.jpg'    

#    master = '/home/filfy/github_clones/sandbar_seg/RC0651R_surveydata/RC0651R_20121009_1219.jpg'
#    slave = '/home/filfy/github_clones/sandbar_seg/RC0651R_surveydata/RC0651R_20140930_1202.JPG'
    
