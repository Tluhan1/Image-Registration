# Area-based approach:
# - Get 2 images (RGB reference image red misaligned image)
# - Get the inverse of the given transform as initial transform
# - Apply the Nelder-Mead simplex method to get new transform

#-----------------------------------------------------------------#

# Dependencies:
import cv2
import sys
import numpy as np
import pandas as pd
from PIL import Image
from scipy import optimize
from matplotlib import pyplot as plt
from IPython.display import clear_output
from skimage.measure import compare_ssim

import warnings
warnings.filterwarnings('ignore')

#-------------------------------------------------------------------------------------------------#

# Read images from file through dataframe referencing:
df = pd.read_pickle("master_manifest_val_red.pkl")
df.drop('survey_id',axis=1,inplace=True)

# Sanity check - valid instance index:
i = int(sys.argv[1],base=10)
if i > len(df):
    print('Invalid argument...')
    sys.exit(1)

I = np.load(df.values[i][2])[0]
I = cv2.normalize(I, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
image_mis = I.astype(np.uint8)
rows,cols = image_mis.shape

# Get reference image (use the red channel):
I = np.load(df.values[i][1])[0][:,:,0]
I = cv2.normalize(I, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
im = Image.fromarray(I.astype(np.uint8))
image_ref = np.array(im.resize((cols,rows)))

#-------------------------------------------------------------------------------------------------#

# Melder-Mead simplex mimization algorithm:
def f(M):
    global image_mis
    global image_ref
    rows,cols = image_mis.shape
    matrix = np.reshape(M, (-1, 3))
    reg_image = cv2.warpAffine(image_mis,matrix,(cols,rows))
    Rev_Corr = 1/np.corrcoef(reg_image.flat, image_ref.flat)[0,1]
    return Rev_Corr

#-------------------------------------------------------------------------------------------------#

# Similarality measures before and after registration:
def display_results(New_matrix,reg_image_new):
    
    print('Transformation matrix:')
    print(New_matrix,'\n')

    (score, diff) = compare_ssim(image_mis, image_ref, full=True)
    diff = (diff * 255).astype("uint8")
    print('Compare reference with misaligned (with the Structural Similarity Index):')
    print("SSIM: {}".format(score),'\n')
    (score1, diff1) = compare_ssim(reg_image_new,image_ref, full=True)    
    diff1 = (diff1 * 255).astype("uint8")
    print('Compare reference with registered (with the Structural Similarity Index):')
    print("SSIM: {}".format(score1))

    Corr = np.corrcoef(image_mis.flat, image_ref.flat)[0,1]
    print('Compare reference with misaligned (with correlation coefficient):')
    print("Correlation coefficient: {}".format(Corr),'\n')
    Corr1 = np.corrcoef(reg_image_new.flat,image_ref.flat)[0,1]
    print('Compare reference with registered (with correlation coefficient):')
    print("Correlation coefficient: {}".format(Corr1))

# Plot images:
def Plot_registration(reg_image_new):
    # Plot registration:
    plt.figure(figsize=(10,12))
    plt.subplot(131)
    plt.imshow(image_ref)
    plt.title('Reference')
    plt.subplot(132)
    plt.imshow(image_mis)
    plt.title('Misaligned')
    plt.subplot(133)
    plt.imshow(reg_image_new)
    plt.title('Registered')
    plt.show()

#-------------------------------------------------------------------------------------------------#
    
# Main
def main():

    # Perform initial transformation: 
    M = np.reshape(df.values[i][3], (-1, 3))
    M = np.append(M,[[0.,0.,1.]],axis=0)
    M_inv = np.linalg.inv(M)
    matrix = np.delete(M_inv,2,0)
    reg_image = cv2.warpAffine(image_mis,matrix,(cols,rows))
    Corr = 1/np.corrcoef(reg_image.flat, image_ref.flat)[0,1]

    # Apply Nelder-Mead algorithm:
    M = matrix.flatten()
    M_corr = optimize.minimize(f,M,method='Nelder-Mead')

    # Perform new transformation and calculate correlation coefficient:
    New_matrix = np.reshape(M_corr.x, (-1, 3))
    reg_image_new = cv2.warpAffine(image_mis,New_matrix,(cols,rows))
    Corr_new = 1/np.corrcoef(reg_image_new.flat, image_ref.flat)[0,1]
    
    # Display results:   
    display_results(New_matrix,reg_image_new)
    
    # Plot results:
    Plot_registration(reg_image_new)

#-------------------------------------------------------------------------------------------------#

if __name__ == "__main__":
    main()