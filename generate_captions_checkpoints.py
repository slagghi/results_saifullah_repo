# This file loads the various checkpoints and generate captions for the whole test set

import json
from helpers import print_progress

image_dir='../../Desktop/UAV/images/'
def bulk_generation(outFile_name):
    filenames_test=load_json('filenames_test')
    path=image_dir
    num_test_images=len(filenames_test)
#    generated_captions=list()
    caption_list=list()
    for i in range(num_test_images):

        C=generate_caption(path+filenames_test[i])
#        generated_captions.append(C)
        caption_list.append(C)
        print_progress(i,num_test_images)
    return caption_list

for i in range(10):
    checkpoint_name=str(i)+'_checkpoint.keras'
    try:
        decoder_model.load_weights(checkpoint_name)
    except Exception as error:
        print('Error trying to load checkpoint.')
        print(error)
    print('\nLoaded model trained on ',i+1,' epochs')
    outFile=str(i)+'_generated_captions_ResNet50_5layers.json'
    caption_list=bulk_generation(outFile)
    with open(outFile,'w') as f:
        json.dump(caption_list,f)
    
    