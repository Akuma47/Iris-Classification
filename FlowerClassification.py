import pandas as pd
import numpy as np
import math

dataset = pd.read_csv('iris.data')


irisData = []
labels = []

for i in range(100):
    x1 = [dataset['x1'][i],dataset['x2'][i],dataset['x3'][i],dataset['x4'][i]]

    irisData.append(x1)

    if dataset['class'][i] == 'Iris-setosa':
        labels.append(0)

    if dataset['class'][i] == 'Iris-versicolor':
        labels.append(0.5)
    
#    if dataset['class'][i] == 'Iris-virginica':
 #       labels.append(1)




epochs = 50
lr = 0.01

w1 = 0.1
w2 = 0.1
w3 = 0.1
w4 = 0.1

loss = 0

for h in range(epochs):

    for i in range(len(irisData)):
        
        x1 = irisData[i][0]
        x2 = irisData[i][1]
        x3 = irisData[i][2]
        x4 = irisData[i][3]
        y = labels[i]

        y_hat = (x1* w1) + (x2 * w2) + (x3 * w3) + (x4 * w4)
        sig = 1 / (1+np.exp(-y_hat))

        loss += - (y*math.log10(sig) + (1-y)*math.log10(1-sig))
        loss = loss/len(irisData)

        # Update weights
        w1 = w1 - (loss * lr * y_hat)
        w2 = w2 - (loss * lr)
        w3 = w3 - (loss * lr * y_hat)
        w4 = w4 - (loss * lr * y_hat)


while True:
    print('')
    x1 = float(input('x1> '))
    x2 = float(input('x2> '))
    x3 = float(input('x3> '))
    x4 = float(input('x4> '))

    res = (x1 * w1) + (x2 * w2) + (x3 * w3) + (x4 * w4)
    sig = 1 / (1+np.exp(-res))

    if sig < 0.5:
        print('Iris Setosa')

    if sig > 0.5:
        print('Versicolar')

  #  if sig > 0.6:
   #     print('Virginica')

    print(sig)























