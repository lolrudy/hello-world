import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as Data
import numpy
import csv
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image
import os
from resnet import resnet34, resnet18
BATCH_SIZE = 32

import random
def most_common(list):
    countdict = {}
    for item in list:
        if item in countdict:
            countdict[item] += 1
        else:
            countdict[item] = 1
    maxcount = 0
    mostcommon = None
    for key in countdict:
        if countdict[key] > maxcount:
            mostcommon = key
            maxcount = countdict[key]
        elif countdict[key] == maxcount:
            if random.randint(0, 1) == 1:
                mostcommon = key
                maxcount = countdict[key]
    return mostcommon

def pred_is_unsure(list):
    countdict = {}
    for item in list:
        if item in countdict:
            countdict[item] += 1
        else:
            countdict[item] = 1
    maxcount = 0
    mostcommon = None
    for key in countdict:
        if countdict[key] > maxcount:
            mostcommon = key
            maxcount = countdict[key]
        elif countdict[key] == maxcount:
            if random.randint(0, 1) == 1:
                mostcommon = key
                maxcount = countdict[key]
    if maxcount <= len(countdict)/2:
        return True
    else:
        return False


train_data = numpy.load("./data/train.npy")
test_data = numpy.load("./data/test.npy")
train_label = []
with open("./data/train.csv") as csvfile:
    csv_reader = csv.reader(csvfile)  # Ê¹ÓÃcsv.reader¶ÁÈ¡csvfileÖÐµÄÎÄ¼þ
    birth_header = next(csv_reader)  # ¶ÁÈ¡µÚÒ»ÐÐÃ¿Ò»ÁÐµÄ±êÌâ
    for row in csv_reader:  # ½«csv ÎÄ¼þÖÐµÄÊý¾Ý±£´æµ½birth_dataÖÐ
        train_label.append(int(row[1]))

train_label = numpy.array(train_label)



from colab import train_model
def load_model(modelclass, modelnum):
    print("loading model "+str(modelnum))
    model = modelclass().cuda()
    path_prefix = "./data/model"
    path_suffix = ".pt"
    path = path_prefix + str(modelnum) + path_suffix
    model.load_state_dict(torch.load(path))
    return model


from colab import MyModel
from googlenet import googlenet
modellist = []
model_num = 4
startnum = 36
from densenet import densenet121, densenet161
for modelnum in [19, 20, 21, 33, 34, 32, 36, 38, 39]:
    #modellist.append(train_model(googlenet, train_data, train_label, 0.92))
    #continue
    if modelnum < 7:
        modellist.append(load_model(MyModel, modelnum))
    elif modelnum < 21:
        modellist.append(load_model(resnet34, modelnum))
    elif modelnum < 28:
        modellist.append(load_model(resnet18, modelnum))
    elif modelnum < 35:
        modellist.append(load_model(googlenet, modelnum))
    elif modelnum != 37:
        modellist.append(load_model(densenet161, modelnum))

from colab import final_test_collate
pred_index = 0
final_test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=final_test_collate)
with open("./data/submit.csv", 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)  # Ê¹ÓÃcsv.reader¶ÁÈ¡csvfileÖÐµÄÎÄ¼þ
    csv_writer.writerow(['image_id', 'label'])
    file = open("./data/unsure.csv", 'w', newline='')
    unsure_writer = csv.writer(file)
    for X in final_test_loader:
        X = X.view(-1, 1, 28, 28).cuda()
        X = Variable(X).float().cuda()
        allpredlist = []
        unsurelist = []
        for i in range(BATCH_SIZE):
            allpredlist.append([])
        for model in modellist:
            test_out = model(X)
            _, pred = test_out.max(1)
            pred_list = pred.cpu().numpy().tolist()
            for index in range(len(pred_list)):
                allpredlist[index].append(pred_list[index])
        finalpredict = []
        for thelist in allpredlist:
            if len(thelist) > 0:
                finalpredict.append(most_common(thelist))
                unsurelist.append(pred_is_unsure(thelist))
        count = 0
        for p in finalpredict:
            csv_writer.writerow([pred_index, p])
            if (unsurelist[count]):
                unsure_writer.writerow([pred_index])
            pred_index = pred_index + 1
            count = count + 1
        
    file.close()



