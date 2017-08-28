
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os
import pandas as pd
from sklearn.externals import joblib
from skimage.color import rgb2gray
from skimage import exposure, feature, transform

if os.path.isfile("../data/clf/clf_svc_hog.pkl"):
    print("[INFO] loading classifier: SVC trained on HoG features...")
    svc = joblib.load("../data/clf/clf_svc_hog.pkl")
    print("[INFO] Classifer is loaded as instance ::svc::")

rootpath = '../data/orl_faces/'
def get_csvfiles(rootpath):
    import re
    csvfiles = []
    root = pathlib.Path(rootpath)
    dirlist = pathlib.os.listdir(root)
    dirs = sorted([x for x in dirlist if re.search(r"^s", x)])
    for i, name in enumerate(dirs):
        for root1, dirs1, files1 in pathlib.os.walk(pathlib.os.path.join(root, name)):
            csvfile = ([pathlib.os.path.join(root, name,x) for x in files1 if re.search(r"^FR", x)])
            csvfiles.append(csvfile[0])
    return csvfiles

def training_data(csvfiles):
    images = []
    labels = []
    for csvfile in csvfiles:
        csvpath = csvfile
        dirpath = '/'.join(csvpath.split('/')[:-1])
        df1 = pd.read_csv(csvpath)
        for im_index in range(0,len(df1)):
            im_path = '/'.join([dirpath,df1.iloc[im_index,0]])
            img = plt.imread(im_path)
            images.append(img)

            lbl = df1.iloc[im_index,1]
            labels.append(lbl)
    return images, labels

csvfiles = get_csvfiles(rootpath)
X, y = training_data(csvfiles)

def testing_on_data():
    n = np.random.randint(0, high=len(y))
    test1 =X[n]
    t1_true = y[n]
    
    fig = plt.figure()
    fig.suptitle('Face recognition', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
    ax.set_title('Person')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    
    grayim = rgb2gray(test1)
    (t1_feat, hogImage) = feature.hog(grayim, orientations=9, pixels_per_cell=(8,8),
        cells_per_block=(2,2), block_norm='L2-Hys', transform_sqrt=True, visualise=True)
#     temp = temp.reshape(1,-1) 
    t1_feat = t1_feat.reshape(1, -1)
    t1_predict = svc.predict(t1_feat)
    print("==========")
    print("True :{}\npredicted:{}\n".format(t1_true,t1_predict[0]))
    # show the prediction
    print("I think this person belongs to class: {}".format(t1_predict[0]))
    print("==========")
    
    ax.set_xlabel('True Class: {}'.format(t1_true))
    ax.set_ylabel('Predicted Class: {}'.format(t1_predict[0]))
    mytext = '{}-{}'.format(t1_true,t1_predict[0])
    if (t1_true==t1_predict[0]):
        ax.text(5, 9.12,mytext, style='italic',bbox={'facecolor':'green', 'alpha':1, 'pad':10}, fontweight='bold')
    else:
        ax.text(5, 9.12, mytext, style='italic',bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
    ax.imshow(test1,cmap='gray')
    plt.show()

for i in range(10):
    testing_on_data()

