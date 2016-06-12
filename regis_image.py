"""
Daniel Buscombe, June 2016
"""

from scipy.misc import imresize, toimage
import imreg_dft as ird
import os
import numpy as np
from scipy.linalg import inv
from PIL import Image

from sift import read_features_from_file, match, process_image
from ransac import doransac, get_points

from fnmatch import filter 

import warnings
warnings.filterwarnings("ignore")

master = '/run/media/dbuscombe/MASTER/GCMRC/SANDBAR_REMOTECAMERAS/RC0307Rf/RC0307Rf_20091012_1130.jpg'

direc = '/run/media/dbuscombe/MASTER/GCMRC/SANDBAR_REMOTECAMERAS/RC0307Rf/'
filenames = sorted(filter(os.listdir(direc), '*.[Jj][Pp][Gg]'))

outdirec = '/run/media/dbuscombe/MASTER/GCMRC/SANDBAR_REMOTECAMERAS/RC0307Rf_regis/'

for slave in filenames:
    try:
        os.mkdir("temp")
    except OSError:
        pass

    im1 = Image.open(slave).convert('L') #sys.argv[1]).convert('L')
    im2 = Image.open(master).convert('L') #sys.argv[2]).convert('L')
        
    w,h = im1.size
        
    im1 = im1.resize((w/4, h/4))    
    im2 = im2.resize((w/4, h/4))    
            
    im1.save('temp/1.pgm')
    im2.save('temp/2.pgm')
    im1 = np.asarray(im1)
    im2 = np.asarray(im2)
    process_image('temp/1.pgm', 'temp/1.key')
    process_image('temp/2.pgm', 'temp/2.key')
    key1 = read_features_from_file('temp/1.key')
    key2 = read_features_from_file('temp/2.key')
    score = match(key1[1], key2[1])
    plist = get_points(key1[0], key2[0], score)

    out = doransac(im1, im2, plist)
    H = inv(out)

    im1 = Image.open(slave)
          
    imtemp = ird.transform_img(np.asarray(im1), tvec=[-H[0][2]*4, -H[1][2]*4])
    
    outfile = outdirec + slave.split(os.sep)[-1].split(os.path.splitext(slave)[1])[0]+'_reg.jpg'
    #os.path.dirname(slave)+os.sep+
    
    toimage(imtemp).save(outfile)
    
    
    ##22mile
    #master = '/home/filfy/github_clones/sandbar_seg/RC0220Ra_surveydata/RC0220Ra_20130922_1353.jpg'
    #slave =  '/home/filfy/github_clones/sandbar_seg/RC0220Ra_surveydata/RC0220Ra_20150924_1141.JPG'

#    master = '/home/filfy/github_clones/sandbar_seg/RC0089L_surveydata/RC0089L_20121004_0959.jpg'
#    slave = '/home/filfy/github_clones/sandbar_seg/RC0089L_surveydata/RC0089L_20130921_1553.jpg'    

#    master = '/home/filfy/github_clones/sandbar_seg/RC0651R_surveydata/RC0651R_20121009_1219.jpg'
#    slave = '/home/filfy/github_clones/sandbar_seg/RC0651R_surveydata/RC0651R_20140930_1202.JPG'
    
