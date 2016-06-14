"""
Daniel Buscombe, June 2016
"""

from random import choice
import numpy as np

from homography import Haffine_from_points

def get_points(locs1, locs2, matchscores):
    '''
        Return the corresponding points in both the images
    '''
    plist = []
    t = min(len(locs1), len(locs2))
    for i in xrange(len(matchscores)):
        if (matchscores[i] > 0):
            y1 = int(locs1[i, 1])
            x1 = int(locs1[i, 0])

            y2 = int(locs2[int(matchscores[i]), 1])
            x2 = int(locs2[int(matchscores[i]), 0])

            plist.append([[x1,y1],[x2,y2]])
    return plist

def doransac(im1, im2, points_list, iters = 10 , error = 10, good_model_num = 5):
    '''
        This function uses RANSAC algorithm to estimate the
        shift and rotation between the two given images
    '''
    
    rows,cols = im1.shape

    model_error = 255
    model_H = None

    for i in xrange(iters):
        consensus_set = []
        points_list_temp = np.copy(points_list).tolist()
        # Randomly select 3 points
        for j in range(3):
            temp = choice(points_list_temp)
            consensus_set.append(temp)
            points_list_temp.remove(temp)
        
        # Calculate the homography matrix from the 3 points
        
        fp0 = []
        fp1 = []
        fp2 = []
        
        tp0 = []
        tp1 = []
        tp2 = []
        for line in consensus_set:
        
            fp0.append(line[0][0])
            fp1.append(line[0][1])
            fp2.append(1)
            
            tp0.append(line[1][0])
            tp1.append(line[1][1])
            tp2.append(1)
            
        fp = np.array([fp0, fp1, fp2])
        tp = np.array([tp0, tp1, tp2])
        
        H = Haffine_from_points(fp, tp)
                            
        # Transform the second image
        # imtemp = transform_im(im2, [-xshift, -yshift], -theta)
        # Check if the other points fit this model

        for p in points_list_temp:
            x1, y1 = p[0]
            x2, y2 = p[1]

            A = np.array([x1, y1, 1]).reshape(3,1)
            B = np.array([x2, y2, 1]).reshape(3,1)
            
            out = B - np.dot(H, A)
            dist_err = np.hypot(out[0][0], out[1][0])
            if dist_err < error:
                consensus_set.append(p)            
            

        # Check how well is our speculated model
        if len(consensus_set) >= good_model_num:
            dists = []
            for p in consensus_set:
                x0, y0 = p[0]
                x1, y1 = p[1]
                
                A = np.array([x0, y0, 1]).reshape(3,1)
                B = np.array([x1, y1, 1]).reshape(3,1)
                
                out = B - np.dot(H, A)
                dist_err = np.hypot(out[0][0], out[1][0])
                dists.append(dist_err)
            if (max(dists) < error) and (max(dists) < model_error):
                model_error = max(dists)
                model_H = H
                        
    return model_H

    
    
    
