import cv2 as cv
import glob
import os
import tqdm
import numpy as np

path = "./valid"
labellist = glob.glob(f"{path}/*GTC.tif")
outPath =  "./valid_label"
if not os.path.exists(outPath):
    os.makedirs(outPath)
for i in tqdm.tqdm(range(len(labellist))):
    label = cv.imread(labellist[i], 0)
    print(np.unique(label))
    labelName = os.path.split(labellist[i])[-1]
    label2 = np.where(label == 224, 0, 255)
    cv.imwrite(f"{outPath}/{labelName}", label2)