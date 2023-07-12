# Load useful library

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import scipy.ndimage
from skimage.color import rgb2hsv, hsv2rgb

# Usefull functions
def setColors(nH, nS, nV):
    """ int**3 -> Array[nH*nS*nV,3]*Array[nH,nS,nV,3]
        computes an RGB palette from a sampling of HSV values
    """
    pal1 = np.zeros((nH*nS*nV, 3))
    pal2 = np.zeros((nH, nS, nV, 3))
    tH, tS, tV = 1/(2*nH), 1/(2*nS), 1/(2*nV)

    idx = 0
    for i in range(nH):
        for j in range(nS):
            for k in range(nV):
                HSVval = np.array([[[i/nH + tH, j/nS + tS, k/nV + tV]]])
                pal1[idx, :] = hsv2rgb(HSVval)*255  
                pal2[i, j, k, :] = hsv2rgb(HSVval)*255
                idx += 1
    return pal1, pal2

def viewQuantizedImage(I,pal):
    """ Array*Array -> Array
        Display an indexed image with colors according to pal 
    """
    Iview = np.empty(I.shape)
    n, m, c = I.shape
    for i in range(n):
        for j in range(m):
            h, s, v = I[i, j, :]
            Iview[i, j, :] = pal[ int(h), int(s), int(v), :]
    print( Iview.max())
    plt.imshow(Iview/255)
    plt.show()

def display5mainColors(histo, pal):
    """ Array*Array -> NoneType
        Display the 5 main colors in histo 
    """
    idx = np.argsort(histo)
    idx = idx[::-1]
    K = 5
    for i in range (K):
        Ia = np.zeros((1, 1, 3), dtype=np.uint8)
        Ia[0,0,0] = pal[idx[i], 0]
        Ia[0,0,1] = pal[idx[i], 1]
        Ia[0,0,2] = pal[idx[i], 2]
        plt.subplot(1, K, i+1)
        plt.imshow(Ia)
        plt.axis('off')
    plt.show()

def display20bestMatches(S, indexQuery):
    """ Array*int -> NoneType 
    """
    L = S[indexQuery, :]
    Idx = np.argsort(L)[::-1]
    cpt = 1
    plt.figure(figsize=(15, 10))
    for idx in Idx[:20]:
        plt.subplot(5, 4, cpt)
        indexQuery = idx
        imageName = (pathImage+listImage[indexQuery]).strip()
        plt.imshow(np.array(Image.open(imageName))/255.)
        plt.title(listImage[indexQuery])
        plt.axis('off')
        cpt += 1
    plt.show()

# that returns the quantize interval of v considering a uniform quantization of values over the range  [0,1]
# with K evenly spaced intervals. For a image value v=1, the function will return K-1.
def quantize(v, k):
    if v==1: return k-1
    return int(v*k)


# Write a function [Iq, histo] = quantizeImage(I,Nh,Ns,Nv) that takes as input one image I of size N x M x 3 in the HSV representation and the number of quantification interval needed for H, S and V. Your function will return:
# Write a function [Iq, histo] = quantizeImage(I,Nh,Ns,Nv) that takes as input one image I of size N x M x 3 in the HSV representation and the number of quantification interval needed for H, S and V. Your function will return:
def quantizeImage(I, nH, nS, nV):
    """ Array*int**3 -> Array*Array
    """
    Iq = np.zeros(I.shape, dtype=int)
    Iq[:,:,0] = [[quantize(j,nH) for j in x] for x in I[:,:,0]]
    Iq[:,:,1] = [[quantize(j,nS) for j in x] for x in I[:,:,1]]
    Iq[:,:,2] = [[quantize(j,nV) for j in x] for x in I[:,:,2]]
    M = np.zeros((nH,nS,nV))
    for x in Iq:
        for h,s,v in x:
            M[h,s,v]+=1
    return Iq, M

def normalize(H):
    return H/np.linalg.norm(H, 2)


def crop_image(I):
    h, w = I.shape[:2]
    position = np.argmax(np.abs(I).mean(2))
    x, y = np.unravel_index(position, I.shape[:2])
    
    if x < 50: x = 50
    elif x>=h-50: x = h-51
    if y < 50: y = 50
    elif y>=w-51: y = w-51

    elu = I[x - 50: x + 50, y - 50: y + 50, :]
    return elu.transpose((2,0,1))

"""
# quantization parameters
nH = 2
nS = 2
nV = 2
# color palette computation
palette, palette2 = setColors(nH, nS , nV );

# Image quantization (your function)
Iq, histo = quantizeImage(J, nH, nS, nV)

# Visualisation of quantized image
viewQuantizedImage(Iq , palette2)

# flat a 3D histogram to 1D
histo = histo.flat

# Histogram normalization (your function)
histo = normalize(histo)

plt.figure()
plt.plot(histo)
plt.show()

## Determine 5 more frequent colors
idx_most_prevalent = (-histo).argsort()[:5]
hsv_most_prevalent = [np.unravel_index(idx,( nH, nS , nV )) for idx in idx_most_prevalent]

display5mainColors(histo, palette)

print(hsv_most_prevalent)

"""