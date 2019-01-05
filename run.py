import numpy as np 
from Train import classifier
import Test
import matplotlib.pyplot as plt
import matplotlib


img = np.load("data_train.npy")
ground = np.load("ground_truth.npy")
train_image = np.load('data_train.npy')

test_image = np.load('data_test.npy')


# plt.imshow(test_image)
# plt.show()

ans = Test.test(test_image)
print("here")
np.savetxt('Test_groudTruth',ans,fmt = '%i')