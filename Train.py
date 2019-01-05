import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix, classification_report

img = np.load("data_train.npy")
ground = np.load("ground_truth.npy")
train_image = np.load('data_train.npy')

coordinates = ground[:, :2]

labels = np.zeros((6250,6250))
# making the coordinates of lables as 1  
for i in range(len(coordinates)):
    x = coordinates[i][0]
    y = coordinates[i][1]
    labels[y][x] = 1

def image_splitter(x,y,size,labels):
    list=[]
    xmin = x - size
    xmax = x + size
    ymin = y - size
    ymax = y + size
    im = train_image[ymin:ymax,xmin:xmax,:]
#     plt.imshow(im)
#     plt.show()
    label_train = labels[ymin:ymax,xmin:xmax]
    return im, label_train

# value for splitting the training image according to regions of interest 
size = 10
i = 0
train_set = []
label_set = []
while i < len(coordinates):
    img,label = image_splitter(coordinates[i][0],coordinates[i][1],size,labels)
    # plt.imshow(img)
    # plt.show()
    train_set.append(img)
    label_set.append(label)
    i = i+1
train_set = np.array(train_set)
label_set = np.array(label_set)

a,b,c,channels = train_set.shape
X = np.reshape(train_set,(a*b*c, channels))
label_set = np.reshape(label_set, (a*b*c))
X_train, X_test, Y_train, Y_test = train_test_split(X,label_set, test_size = 0.3, random_state=50)

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, Y_train)
classifier.score(X_test, Y_test)

cv_results = cross_validate(classifier, X, label_set, cv=10, return_train_score=False)
sorted(cv_results.keys())                         
cv_results['test_score']
# im = train_image[400:800,:1000,:]
# plt.imshow(im)
pred = classifier.predict(X_test)
np.count_nonzero(pred)
results = confusion_matrix(Y_test,pred)
# print(classification_report(Y_test,pred))
# print(results)

# a,b,c = test_image.shape
# test_data = np.reshape(test_image,(a*b,c))
# predf = classifier.predict(test_data)
