import mxnet as mx
from mxnet import image, gpu
from mxnet.gluon.data.vision import transforms
import gluoncv
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from gluoncv.data.transforms.presets.segmentation import test_transform
from gluoncv.utils.viz import get_color_pallete
import matplotlib.image as mpimg

#ctx = mx.cpu(0)
ctx = mx.gpu(0)

df = pd.DataFrame()

newlocation = pd.read_csv('D:/python/images/newlocation.csv', index_col=0)
path_base = r"D:\python\images//"

'''
path = r"D:\python\images\1.338012,103.817459//"
filename = path + "&heading=0.0&pitch=0.0.jpg"
path_out = r"D:\python\mask\1.338012,103.817459"
file_out = path_out + "//" + "&heading=0.0&pitch=0.0.png"
'''

orientation_set = ["&heading=0.0&pitch=0.0",
                  "&heading=90.0&pitch=0.0",
                  "&heading=180.0&pitch=0.0",
                  "&heading=270.0&pitch=0.0",]

model = gluoncv.model_zoo.get_model('deeplab_v3b_plus_wideresnet_citys', pretrained=True)
model.collect_params().reset_ctx(ctx=mx.gpu())



def ImgSeg(filename, model):
    img = image.imread(filename)
    plt.imshow(img.asnumpy())
    #plt.show()
    img = test_transform(img, ctx)

    output = model.predict(img)
    predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
    return predict


def GetLabel(predict):
    label=[]
    for i in range(len(predict[:,0])):
        for j in range(len(predict[0,:])):
            if (model.classes[int(predict[i,j])] not in label):
                label.append(model.classes[int(predict[i,j])])
    print("label",label)

def GetRatio(predict):
    #calculating the ratio of different features
    Ratio={}
    total_pixel = len(predict[:,0]) * len(predict[0,:])
    for i in range(len(predict[:,0])):
        for j in range(len(predict[0,:])):
            if (model.classes[int(predict[i,j])] not in Ratio.keys()):
                Ratio[model.classes[int(predict[i,j])]] = 1/total_pixel
            else:
                Ratio[model.classes[int(predict[i,j])]] = (Ratio[model.classes[int(predict[i,j])]]*total_pixel+ 1)/total_pixel

    return Ratio


def save_mask(path_out, file_out, predict):
    mask = get_color_pallete(predict, 'cityscapes')

    if not os.path.exists(path_out):  # create filefolder of images
        os.makedirs(path_out)

    #print(file_out)
    mask.save(file_out)
    mmask = mpimg.imread(file_out)
    plt.imshow(mmask)
    #plt.show()


for i in range(len(newlocation))[0:10]:
    print(i)
    k = 0
    for j in orientation_set:

        filename = path_base + str(newlocation["lat"][i]) +"," + str(newlocation["lng"][i]) + "//" + j + ".jpg"
        path_out = r"D:\python\mask//" + str(i) + "," + str(newlocation["lat"][i]) +"," + str(newlocation["lng"][i])
        file_out = r"D:\python\mask//" + str(i) + "," + str(newlocation["lat"][i]) +"," + str(newlocation["lng"][i]) + "//" + j + ".png"

        #image segmentation
        predict = ImgSeg(filename = filename, model = model)

        #calculate ratio
        Ratio = GetRatio(predict)
        location = dict(newlocation.loc[i])
        location["orientation"] = j

        Merged = dict(list(location.items()) + list(Ratio.items()))
        df.loc[i * 4 + k, "Point_ID"] = i
#        df["Point_ID"][(i-1)*4+j] = i
        df = df.append(Merged, ignore_index=True)
        save_mask(path_out = path_out, file_out =file_out, predict = predict)
        k = k + 1


df.to_csv('D:/python/df.csv')

