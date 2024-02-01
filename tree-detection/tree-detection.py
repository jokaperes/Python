from PIL import Image
import numpy as np
from skimage import data
from skimage.feature import match_template
import matplotlib.pyplot as plt

imageTotal = np.asarray(Image.open('images/OlivoTotal.png'))
imageSmall = np.asarray(Image.open('images/OlivoTemplate_small.png'))
imageMedium = np.asarray(Image.open('images/OlivoTemplate_medium.png'))
imageLarge = np.asarray(Image.open('images/OlivoTemplate_large.png'))
imageExtra = np.asarray(Image.open('images/OlivoTemplate_extra.png'))

image = imageTotal[:,:,1]
treeSmall = imageSmall[:,:,1]
treeMedium = imageMedium[:,:,1]
treeLarge = imageLarge[:,:,1]
treeExtra = imageExtra[:,:,1]

fig = plt.figure(figsize = (8, 1.8))
ax1 = plt.subplot(1,4,1)
ax2 = plt.subplot(1,4,2)
ax3 = plt.subplot(1,4,3)
ax4 = plt.subplot(1,4,4)
ax1.imshow(treeSmall, cmap = plt.cm.gray)
ax1.set_title('small')
ax2.imshow(treeMedium, cmap = plt.cm.gray)
ax2.set_title('medium')
ax3.imshow(treeLarge, cmap = plt.cm.gray)
ax3.set_title('large')
ax4.imshow(treeExtra, cmap = plt.cm.gray)
ax4.set_title('extra')

resultSmall = match_template(image, treeSmall)
resultSmallFilter = np.where(resultSmall > 0.85)

resultMedium = match_template(image, treeMedium)
resultMediumFilter = np.where(resultMedium > 0.95)

resultLarge = match_template(image, treeLarge)
resultLargeFilter = np.where(resultLarge > 0.95)

resultExtra = match_template(image, treeExtra)
resultExtraFilter = np.where(resultExtra > 0.95)

def listaPontos(result):
  xlist = []
  ylist = []
  for ponto in range(np.shape(result)[1]):
    xlist.append(result[1][ponto])
    ylist.append(result[0][ponto])
  return xlist, ylist

plt.plot(listaPontos(resultSmallFilter)[0], listaPontos(resultSmallFilter)[1], 'o', markeredgecolor = 'g',
markerfacecolor = 'none', markersize = 10, label = 'Small')

plt.plot(listaPontos(resultMediumFilter)[0], listaPontos(resultMediumFilter)[1], 'o', markeredgecolor = 'r',
markerfacecolor = 'none', markersize = 10, label = 'Medium')

plt.plot(listaPontos(resultLargeFilter)[0], listaPontos(resultLargeFilter)[1], 'o', markeredgecolor = 'b',
markerfacecolor = 'none', markersize = 10, label = 'Large')

plt.plot(listaPontos(resultExtraFilter)[0], listaPontos(resultExtraFilter)[1], 'o', markeredgecolor = 'y',
markerfacecolor = 'none', markersize = 10, label = 'Extra')

plt.imshow(imageTotal[10: -10, 10:, :])