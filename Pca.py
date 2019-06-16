# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 22:08:45 2019

@author: user
"""

import cv2
import glob

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

names=["SUNDAR","BILL","ELON","JEFF"]

from numpy import array
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

#images = [cv2.imread(file) for file in glob.glob("E:/Bel Internship/path/pins_Sundar Pichai/Sundar Pichai2_740.jpg")]
#filelist = glob.glob('E:\Bel Internship\pins_Sundar Pichai\*.jpg')
#images_sundar= np.array([np.array(Images.open(fname))])


images_sundar = np.array([cv2.imread(file,0) for file in glob.glob("E:\Bel Internship\pins_Sundar Pichai\*.jpg")])

images_sundar.shape
type(images_sundar[0])

for i in range(89):
    images_sundar[i] = np.array(cv2.resize(images_sundar[i],(299,299)))

# =============================================================================
# cv2.imshow('asdqw',images_sundar[73])
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# 
# =============================================================================


images_bill = np.array([cv2.imread(file,0) for file in glob.glob("E:\Bel Internship\pins_bill gates\*.jpg")])
images_bill.shape
bill_vector = np.zeros(shape=(images_bill.shape[0],images_bill.shape[1]*images_bill.shape[2]))

for i in range(86):
    bill_vector[i] = images_bill[i].reshape(1,299*299)

plt.imshow(images_sundar[87].reshape(299,299),cmap=plt.cm.bone)



images_musk = np.array([cv2.imread(file,0) for file in glob.glob("E:\Bel Internship\pins_elon musk\*.jpg")])

images_musk.shape
type(images_musk[0])

for i in range(85):
    images_musk[i] = np.array(cv2.resize(images_musk[i],(299,299)))

plt.imshow(images_musk[84].reshape(299,299),cmap=plt.cm.bone)



images_jeff = np.array([cv2.imread(file,0) for file in glob.glob("E:\Bel Internship\pins_jeff bezos\*.jpg")])

images_jeff.shape
type(images_jeff[0])

for i in range(88):
    images_jeff[i] = np.array(cv2.resize(images_jeff[i],(299,299)))

plt.imshow(images_jeff[87].reshape(299,299),cmap=plt.cm.bone)

fig = plt.figure(figsize=(10,6))
for i in range(88):
    ax=fig.add_subplot(8,11,i+1,xticks=[],yticks=[])
    ax.imshow(images_jeff[i].reshape(299,299),cmap=plt.cm.bone)



























id_sundar=[0]*70
id_bill=[1]*70
id_train=[0]*79 + [1]*76 + [2]*75 + [3]*78
 
sundar_vector= np.zeros(shape=(89,299*299))
jeff_vector = np.zeros(shape=(88,299*299))
musk_vector = np.zeros(shape=(85,299*299))


for i in range(89):
    sundar_vector[i] = images_sundar[i].reshape(1,299*299)
    if i<85:
        musk_vector[i] = images_musk[i].reshape(1,299*299)
    if i<88:
        jeff_vector[i] = images_jeff[i].reshape(1,299*299)
    
plt.imshow(musk_vector[84].reshape(299,299),cmap=plt.cm.bone)

bill_train= bill_vector[:76]
bill_test=bill_vector[76:]

sundar_train=sundar_vector[:79]
sundar_test=sundar_vector[79:]


musk_train=musk_vector[:75]
musk_test=musk_vector[75:]


jeff_train=jeff_vector[:78]
jeff_test=jeff_vector[78:]


id_test= [0]*10 + [1]*10 + [2]*10 + [3]*10

X_train=np.concatenate((sundar_train,bill_train,musk_train,jeff_train),axis=0)

X_test= np.concatenate((sundar_test,bill_test,musk_test,jeff_test),axis=0)

    
#plt.imshow(X_train[80].reshape(299,299),cmap=plt.cm.bone)


from sklearn import decomposition
pca = decomposition.PCA(n_components=250, whiten=True)
pca.fit(X_train)


plt.imshow(pca.mean_.reshape(299,299),cmap=plt.cm.bone)

print(pca.components_.shape)

# =============================================================================
# 
# fig = plt.figure(figsize=(16, 6))
# for i in range(25):
#     ax = fig.add_subplot(5, 5, i + 1, xticks=[], yticks=[])
#     ax.imshow(pca.components_[i].reshape(299,299),cmap=plt.cm.bone)
# 
# =============================================================================
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print(X_train_pca.shape)


from sklearn import svm
clf = svm.SVC()
clf.fit(X_train_pca, id_train)


y_pred=clf.predict(X_test_pca)


fig = plt.figure(figsize=(8, 6))
fig.suptitle('Components 250', fontsize=16)
for i in range(40):
    ax = fig.add_subplot(5, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(X_test[i].reshape(299,299),cmap=plt.cm.bone)
    color = ('green' if y_pred[i] == id_test[i] else 'red')
    ax.set_title(names[y_pred[i]],fontsize="small",color=color)
    
# =============================================================================
#     color = ('black' if y_pred == id_test[i] else 'red')
#     ax.set_title(y_pred[i],fontsize='small', color=color)
# 
# =============================================================================



fig = plt.figure(figsize=(10, 6))
for i in range(80):
    ax = fig.add_subplot(8, 10, i + 1, xticks=[], yticks=[])
    ax.imshow(bill_vector[i].reshape(299,299),cmap=plt.cm.bone)
