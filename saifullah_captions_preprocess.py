# This code takes the saifullah captions
# and saves them in the same manner as
# the RSICD captions

import json
from helpers import load_json

train_json='dataset/Train_caption_new.json'
test_json='dataset/Test_caption_new.json'
val_json='dataset/Validation_caption_new.json'

with open(train_json,'r') as inFile:
    data_train=json.load(inFile)
train_filenames_saifullah=data_train['fileName']
num_train_images=len(train_filenames_saifullah)

train_captions_saifullah=list()
for i in range(num_train_images):
#    Get the 5 raw sentences
    raw=data_train['images'][i]['sentences']
    caps_per_image=len(raw)
    cap_list=list()
    for j in range(caps_per_image):
        c=raw[j]['raw']
        cap_list.append(c)
    train_captions_saifullah.append(cap_list)
    

with open(test_json,'r') as inFile:
    data_test=json.load(inFile)
test_filenames_saifullah=data_test['fileName']
num_test_images=len(test_filenames_saifullah)

test_captions_saifullah=list()
for i in range(num_test_images):
#    Get the 5 raw sentences
    raw=data_test['images'][i]['sentences']
    caps_per_image=len(raw)
    cap_list=list()
    for j in range(caps_per_image):
        c=raw[j]['raw']
        cap_list.append(c)
    test_captions_saifullah.append(cap_list)
    
    
with open(val_json,'r') as inFile:
    data_val=json.load(inFile)
val_filenames_saifullah=data_val['fileName']
num_val_images=len(val_filenames_saifullah)

val_captions_saifullah=list()
for i in range(num_val_images):
#    Get the 5 raw sentences
    raw=data_val['images'][i]['sentences']
    caps_per_image=len(raw)
    cap_list=list()
    for j in range(caps_per_image):
        c=raw[j]['raw']
        cap_list.append(c)
    val_captions_saifullah.append(cap_list)

# They must be saved as tuples
train_captions_saifullah=tuple(train_captions_saifullah)
test_captions_saifullah=tuple(test_captions_saifullah)
val_captions_saifullah=tuple(val_captions_saifullah)

# save to file
with open('dataset/captions_train_saifullah.json','w') as outfile:
    json.dump(train_captions_saifullah,outfile)
    
with open('dataset/captions_test_saifullah.json','w') as outfile:
    json.dump(test_captions_saifullah,outfile)
    
with open('dataset/captions_val_saifullah.json','w') as outfile:
    json.dump(val_captions_saifullah,outfile)
    
with open('dataset/filenames_train_saifullah.json','w') as outfile:
    json.dump(train_filenames_saifullah,outfile)
with open('dataset/filenames_test_saifullah.json','w') as outfile:
    json.dump(test_filenames_saifullah,outfile)
with open('dataset/filenames_val_saifullah.json','w') as outfile:
    json.dump(val_filenames_saifullah,outfile)
    