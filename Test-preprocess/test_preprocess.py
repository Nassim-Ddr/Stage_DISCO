import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy import signal

img_nr = mpimg.imread("../data/red_eyes/18.jpg")
img_r  = mpimg.imread("../data/red_eyes/18r.jpg")

diff = np.abs(img_nr - img_r)
print(diff.shape)

diff_nb = diff.max(2)

diff_p = np.where(diff_nb > 50, 255, 0)

mean_ker = np.ones((16,16)) * 1/256

diff_p = signal.convolve2d(diff_p, mean_ker, boundary='symm', mode='same')

diff_p = np.where(diff_p > 200, 255, 0)



###############################################################

h, w = diff_p.shape

# while (h > 250  and w > 250) : 
#     sub_1 = diff_p[0:h//2, 0:w//2]
#     sub_2 = diff_p[h//2:h, w//2:w]
#     sub_3 = diff_p[0:h//2, w//2:w]
#     sub_4 = diff_p[h//2:h, 0:w//2]

#     l = [sub_1, sub_2, sub_3, sub_4]
#     m = [i.mean() for i in l]
#     diff_p = l[np.argmax(m)]

#     h, w = diff_p.shape
###############################################################
#METHODE POUR TESTER RAPIDEMENT LES YEUX ROUGES 
e = np.argmax(diff_p)
x, y = np.unravel_index(e, diff_p.shape)
elu = diff_p[x - 100: x + 100, y - 100: y + 100]

imgplot = plt.imshow(elu)
###############################################################


# imgplot = plt.hist(diff_p)
# print(np.unique(diff_p))

plt.show()