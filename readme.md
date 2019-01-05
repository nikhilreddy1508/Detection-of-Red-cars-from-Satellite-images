# Detection of Red Cars from High Definition Satellite Imagery   
Red Car detection in satellite images using KNN Classifier and Probabilistic Generative Classifier with RGB values of every pixel as features to the classifiers. 

To reduce the computational time of training all the pixels in the satellite images, preprocessing step was added by constructing sub images based on regions of interest and training the classifier with these sub images

# How to Run:
```
python run.py
```

# How to change parameters

In line number 38 for Train.py we can give the size into which each  training image should be split i.e. for m = 20, we get a 40*40 image of red car.

In line number 57 we can change the number of neighbors of knn algorithm

# Packages Required

Anaconda
