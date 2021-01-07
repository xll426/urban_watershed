import numpy as np
import tqdm

a = np.random.randint(0,10,(300,2000))
b = np.random.randint(0,10,(300,1000))
rowA,widthA = a.shape
rowB,widthB = b.shape

###broadcast
results = []
for i in tqdm.tqdm(range(widthB)):
    b1 = b[:,i].reshape(rowB,1)
    temp = np.arange(widthA)*0
    b1 = b1+temp
    EM = (b1-a)*(b1-a)
    results.append(EM)

####one2one
results2 = np.ones((rowA,widthA,widthB))
for i in tqdm.tqdm(range(rowA)):
    for j in range(widthA):
        for k in range(widthB):
           results2[i,j,k] = (a[i,j]-b[i,k])*(a[i,j]-b[i,k])