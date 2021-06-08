import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# On charge les images
ng_2000 = cv.imread('amazon_2000.png', 0)
ng_2012 = cv.imread('amazon_2012.png', 0)

# On verifie leur taille
print(ng_2000.shape, ng_2012.shape)

blur = cv.GaussianBlur(ng_2000, (5, 5), 0)
ret3, bin_2000 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
blur = cv.GaussianBlur(ng_2012, (5, 5), 0)
ret3, bin_2012 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)


# affichage des images en noir et blanc
fig = plt.figure(figsize=(8, 8))

fig.add_subplot(1, 2, 1)
plt.imshow(bin_2000, cmap="gray")

fig.add_subplot(1, 2, 2)
plt.imshow(bin_2012, cmap="gray")
plt.show()

# On recupere nos elements structurants:
kernel_cross = cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))
kernel_rect = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
kernel_ellipse = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))


"""On choisira d'appliquer d'abord une ouverture puis une fermeture
car c'est la technique utilisée puur débruiter les empruntes digitales
et que notre foret y ressemble"""

#Cross kernel
denoised_cross_bin_2000 = cv.morphologyEx(cv.morphologyEx(
	bin_2000, cv.MORPH_OPEN, kernel_cross),cv.MORPH_CLOSE, kernel_cross)
"""plt.imshow(denoised_cross_bin_2000,cmap="gray")
plt.show()"""
denoised_cross_bin_2012 = cv.morphologyEx(cv.morphologyEx(
	bin_2012, cv.MORPH_OPEN, kernel_cross),cv.MORPH_CLOSE, kernel_cross)

#Rectangle kernel
denoised_rect_bin_2000 = cv.morphologyEx(cv.morphologyEx(
	bin_2000, cv.MORPH_OPEN, kernel_cross),cv.MORPH_CLOSE, kernel_rect)
denoised_rect_bin_2012 = cv.morphologyEx(cv.morphologyEx(
	bin_2012, cv.MORPH_OPEN, kernel_cross),cv.MORPH_CLOSE, kernel_rect)


#Ellipse Kernel
denoised_ellipse_bin_2000 = cv.morphologyEx(cv.morphologyEx(
	bin_2000, cv.MORPH_OPEN, kernel_cross),cv.MORPH_CLOSE, kernel_ellipse)
denoised_ellipse_bin_2012 = cv.morphologyEx(cv.morphologyEx(
	bin_2012, cv.MORPH_OPEN, kernel_cross),cv.MORPH_CLOSE, kernel_ellipse)



fig2 = plt.figure(figsize=(8, 12))

fig2.add_subplot(1, 3, 1)
plt.imshow(denoised_cross_bin_2012, cmap="gray")
plt.title('denoised_cross_bin_2012')

fig2.add_subplot(1, 3, 2)
plt.imshow(denoised_rect_bin_2012, cmap="gray")
plt.title('denoised_rect_bin_2012')

fig2.add_subplot(1, 3, 3)
plt.imshow(denoised_ellipse_bin_2012, cmap="gray")
plt.title('denoised_ellipse_bin_2012')

plt.show()


#On choisira ellipse kernel pour comparer la deforestation
num_zeros_2000 = (denoised_ellipse_bin_2000 == 0).sum()
num_ones_2000 = (denoised_ellipse_bin_2000 == 255).sum()
deforest_rate_2000=100*num_ones_2000/num_zeros_2000
print(num_zeros_2000,num_ones_2000)
print('deforest_rate_2000=',deforest_rate_2000,'%')

num_zeros_2012 = (denoised_ellipse_bin_2012 == 0).sum()
num_ones_2012 = (denoised_ellipse_bin_2012 == 255).sum()
deforest_rate_2012=100*num_ones_2012/num_zeros_2012
print(num_zeros_2012,num_ones_2012)
print('deforest_rate_2012=',deforest_rate_2012,'%')



fig = plt.figure(figsize=(8, 8))

fig.add_subplot(1, 2, 1)
plt.imshow(denoised_ellipse_bin_2000, cmap="gray")
plt.title('deforest_rate_2000='+str(round(deforest_rate_2000))+'%')

fig.add_subplot(1, 2, 2)
plt.imshow(denoised_ellipse_bin_2012, cmap="gray")
plt.title('deforest_rate_2012='+str(round(deforest_rate_2012))+'%')

plt.show()