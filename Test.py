from Train import classifier
import numpy as np

def test(test_image):
    a,b,c = test_image.shape
    test_data = np.reshape(test_image,(a*b,c))
    
    predf = classifier.predict(test_data)
    
    final = []
    i=0
    while(i < len(predf)):
        val = []
        if(predf[i] == 1):
            x_coordinate = int(i/len(test_image))
            y_coordinate = (i%len(test_image))
            val.append(x_coordinate)
            val.append(y_coordinate)
            final.append(val)
        i = i+1
    return final
