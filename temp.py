
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
'''
print(math.floor(1.5))
for inception_weight in np.linspace(0,1,endpoint=False):
    for resnet_weight in np.linspace(0,1,endpoint=False):
        inception_weight = round(inception_weight,2)
        resnet_weight = round(resnet_weight,2)
        if inception_weight+resnet_weight>=round(1,2):
            continue
        else:
            mobilenet_weight = round(1-inception_weight-resnet_weight,2)
'''
x =  np.arange(1,13)
y = []
for a in np.arange(0.5,1,0.01):
    a = np.around(a,2)
    y = np.power(x,np.ones(x.shape[0])*(1-a))/(1-a)
    plt.plot(range(len(y)), y, "-*",label=str(a))
#plt.plot(range(len(y)),range(len(y)),"-D")

plt.legend(loc="upper left",ncol=10)
plt.title("model utility with various alpha")
plt.show()

