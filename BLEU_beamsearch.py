# this code evaluates the captions generated using beamsearch
import json
import nltk
import math
from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from helpers import load_json
from helpers import print_progress
from helpers import load_image
import numpy as np
import copy
img_size=(228,228)


from Scores import consensus_score
chencherry=SmoothingFunction()


# LOAD THE TRANSFER MODEL
# can do thi only at first execution
#
#from tensorflow.python.keras.applications import VGG16
#from tensorflow.python.keras.models import Model
#image_model = VGG16(include_top=True, weights='imagenet')
#transfer_layer=image_model.get_layer('fc2')
#image_model_transfer = Model(inputs=image_model.input,
#                             outputs=transfer_layer.output)

transfer_values_train=np.load('image_features/transfer_values/InceptionV3/transfer_values_train.npy')
transfer_values_test=np.load('image_features/transfer_values/InceptionV3/transfer_values_test.npy')
captions_train=load_json('captions_train')
filename='InceptionCaptions/5_beamsearched.json'

out_dir='best_beamsearched/InceptionV3/'

with open(filename,'r') as inFile:
    beamCaptions=json.load(inFile)
    beamCaptions=tuple(beamCaptions)

def get_transfer_values(image_path):
    tv_len=transfer_values_test[0].shape[0]
    filename=image_path[len(image_dir):]
    for i in range(len(filenames_test)):
        if filenames_test[i] == filename:
            break
    transfer_values=transfer_values_test[i]
    transfer_values=np.reshape(transfer_values,(1,tv_len))
    return transfer_values


captions_test=load_json('captions_test_saifullah')
filenames_test=load_json('filenames_test_saifullah')
captions_train=load_json('captions_train_saifullah')

# build the references library
num_samples=len(captions_test)
references=list()
for i in range(num_samples):
    R=captions_test[i]
    refList=list()
    for j in range(len(captions_test[0])):
        refList.append(nltk.word_tokenize(R[j]))
    references.append(refList)
    
print('Choosing best caption based on blue score (cheating)')
bestCaptions=list()
idxList=list()
for i in range(num_samples):
    bleuSums=list()
    for j in range(5):
        candidate=beamCaptions[i][j]['sentence']
        candidate_tokenized=nltk.word_tokenize(candidate)

        s1=sentence_bleu(references[i],candidate_tokenized,weights=[1,0,0,0],smoothing_function=chencherry.method1)
        s2=sentence_bleu(references[i],candidate_tokenized,weights=[0.5,0.5,0,0],smoothing_function=chencherry.method1)
        s3=sentence_bleu(references[i],candidate_tokenized,weights=[0.33,0.33,0.33,0],smoothing_function=chencherry.method1)
        s4=sentence_bleu(references[i],candidate_tokenized,weights=[0.25,0.25,0.25,0.25],smoothing_function=chencherry.method1)

# First approach: the best sentence is the one with the best
# BLEU sum
        bleuSum=s1+s2+s3+s4
        bleuSums.append(bleuSum)
    bestIdx=np.argmax(bleuSums)
#    print(bestCaption,bleuSums)
    bestCaption=beamCaptions[i][bestIdx]['sentence']
    bestCaptions.append(bestCaption[1:])
    idxList.append(bestIdx)
#    plot progress
    print_progress(i+1,num_samples)
    
with open(out_dir+'absolute_best.json','w') as outFile:
    json.dump(bestCaptions,outFile)


# SECON METHOD
# the best caption is the one with the best consensus score
    
# VGG16 consensus
    
print('\nChoosing best caption based on VGG16 consensus')
image_dir='UAV/images/'
VGG16_idxList=list()
VGG16_bestCaptions=list()

transfer_values_train=np.load('image_features/transfer_values/VGG16/transfer_values_train.npy')
transfer_values_test=np.load('image_features/transfer_values/VGG16/transfer_values_test.npy')


for i in range(num_samples):
    consScores=list()
    image_filename=filenames_test[i]
    transfer_values=get_transfer_values(image_dir+image_filename)
    
    for j in range(5):
        candidate=beamCaptions[i][j]['sentence']

        score=consensus_score(candidate,transfer_values,transfer_values_train)
        consScores.append(score)
    bestIdx=np.argmax(consScores)
    VGG16_idxList.append(bestIdx)    
    bestCaption=beamCaptions[i][bestIdx]['sentence']
    VGG16_bestCaptions.append(copy.copy(bestCaption[1:]))

    print_progress(i+1,num_samples)
with open(out_dir+'VGG16_consensus.json','w') as outFile:
    json.dump(VGG16_bestCaptions,outFile)

# ResNet consensus
print('\nChoosing best caption based on ResNet50 consensus')
image_dir='UAV/images/'
ResNet50_idxList=list()
ResNet50_bestCaptions=list()

transfer_values_train=np.load('image_features/transfer_values/ResNet50/transfer_values_train.npy')
transfer_values_test=np.load('image_features/transfer_values/ResNet50/transfer_values_test.npy')


for i in range(num_samples):
    consScores=list()
    image_filename=filenames_test[i]
    transfer_values=get_transfer_values(image_dir+image_filename)
    
    for j in range(5):
        candidate=beamCaptions[i][j]['sentence']

        score=consensus_score(candidate,transfer_values,transfer_values_train)
        consScores.append(score)
    bestIdx=np.argmax(consScores)
    ResNet50_idxList.append(bestIdx)    
    bestCaption=beamCaptions[i][bestIdx]['sentence']
    ResNet50_bestCaptions.append(copy.copy(bestCaption[1:]))

    print_progress(i+1,num_samples)
with open(out_dir+'ResNet50_consensus.json','w') as outFile:
    json.dump(ResNet50_bestCaptions,outFile)

# Inception consensus
print('\nChoosing best caption based on Inception consensus')
image_dir='UAV/images/'
InceptionV3_idxList=list()
InceptionV3_bestCaptions=list()

transfer_values_train=np.load('image_features/transfer_values/InceptionV3/transfer_values_train.npy')
transfer_values_test=np.load('image_features/transfer_values/InceptionV3/transfer_values_test.npy')


for i in range(num_samples):
    consScores=list()
    image_filename=filenames_test[i]
    if i==884:
        image_filename=filenames_test[883]
    transfer_values=get_transfer_values(image_dir+image_filename)
    
    for j in range(5):
        candidate=beamCaptions[i][j]['sentence']

        score=consensus_score(candidate,transfer_values,transfer_values_train)
        consScores.append(score)
    bestIdx=np.argmax(consScores)
    InceptionV3_idxList.append(bestIdx)    
    bestCaption=beamCaptions[i][bestIdx]['sentence']
    InceptionV3_bestCaptions.append(copy.copy(bestCaption[1:]))

    print_progress(i+1,num_samples)
with open(out_dir+'InceptionV3_consensus.json','w') as outFile:
    json.dump(InceptionV3_bestCaptions,outFile)
    