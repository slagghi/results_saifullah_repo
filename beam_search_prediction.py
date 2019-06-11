from helpers import load_image
import numpy as np
from helpers import load_json
import copy

transfer_values_train=np.load('dataset/transfer_values_train_saifullah.npy')
transfer_values_test=np.load('dataset/transfer_values_test_saifullah.npy')
#transfer_values_val=np.load('dataset/transfer_values_val_saifullah.npy')

# This code implements beam search for a less-greedy sentence generation

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def generate_caption(image_path, max_tokens=30):
    """
    Generate a caption for the image in the given path.
    The caption is limited to the given number of tokens (words).
    """

    # Load and resize the image.
    image = load_image(image_path, size=img_size)
    
    # Expand the 3-dim numpy array to 4-dim
    # because the image-model expects a whole batch as input,
    # so we give it a batch with just one image.
    image_batch = np.expand_dims(image, axis=0)

    # Process the image with the pre-trained image-model
    # to get the transfer-values.
    transfer_values = image_model_transfer.predict(image_batch)

    # Pre-allocate the 2-dim array used as input to the decoder.
    # This holds just a single sequence of integer-tokens,
    # but the decoder-model expects a batch of sequences.
    shape = (1, max_tokens)
    decoder_input_data = np.zeros(shape=shape, dtype=np.int)

    # The first input-token is the special start-token for 'ssss '.
    token_int = token_start

    # Initialize an empty output-text.
    output_text = ''

    # Initialize the number of tokens we have processed.
    count_tokens = 0

    # While we haven't sampled the special end-token for ' eeee'
    # and we haven't processed the max number of tokens.
    while token_int != token_end and count_tokens < max_tokens:
        # Update the input-sequence to the decoder
        # with the last token that was sampled.
        # In the first iteration this will set the
        # first element to the start-token.
        decoder_input_data[0, count_tokens] = token_int

        # Wrap the input-data in a dict for clarity and safety,
        # so we are sure we input the data in the right order.
        x_data = \
        {
            'transfer_values_input': transfer_values,
            'decoder_input': decoder_input_data
        }

        # Note that we input the entire sequence of tokens
        # to the decoder. This wastes a lot of computation
        # because we are only interested in the last input
        # and output. We could modify the code to return
        # the GRU-states when calling predict() and then
        # feeding these GRU-states as well the next time
        # we call predict(), but it would make the code
        # much more complicated.
        
        # Input this data to the decoder and get the predicted output.
        decoder_output = decoder_model.predict(x_data)

        # Get the last predicted token as a one-hot encoded array.
        # Note that this is not limited by softmax, but we just
        # need the index of the largest element so it doesn't matter.
        token_onehot = decoder_output[0, count_tokens, :]

        # Convert to an integer-token.
        token_int = np.argmax(token_onehot)

        # Lookup the word corresponding to this integer-token.
        sampled_word = tokenizer.token_to_word(token_int)

        # Append the word to the output-text.
        output_text += " " + sampled_word

        # Increment the token-counter.
        count_tokens += 1

    # This is the sequence of tokens output by the decoder.
    output_tokens = decoder_input_data[0]

    # Plot the image.
#    plt.imshow(image)
#    plt.title(output_text.replace(" eeee",""))
#    plt.axis('off')
#    plt.show()
#    plt.savefig("test_results/test.png", bbox_inches='tight')
    
    # Print the predicted caption.
#    print("Predicted caption:")
#    print(output_text.replace(" eeee",""))
#    print()
    return output_text.replace(" eeee","")

mark_start='ssss '
mark_end=' eeee'
captions_train=load_json('captions_train')
captions_train_marked=mark_captions(captions_train)
captions_train_flat=flatten(captions_train_marked)
tokenizer=TokenizerWrap(texts=captions_train_flat,
                        num_words=2000)

token_start=tokenizer.word_index[mark_start.strip()]
token_end=tokenizer.word_index[mark_end.strip()]

# ASSUME I ALREADY HAVE THE TRANSFER VALUES FOR THE IMAGE
filenames_test=load_json('filenames_test_saifullah')
path='../../../../Desktop/UAV/images/'
filename=filenames_test[812]
image = load_image(path+filename, size=img_size)
image_batch = np.expand_dims(image, axis=0)
transfer_values = image_model_transfer.predict(image_batch)

# This code, given a transfer vector and the previous sequence, predicts the
# K best next words (start with 2)
prev_sequence=[]

def predict_next_word(transfer_values,prev_sequence):
    count_tokens=len(prev_sequence)
    shape=(1,30)
    decoder_input_data = np.zeros(shape=shape, dtype=np.int)
    for i in range(count_tokens):
        decoder_input_data[0,i]=prev_sequence[i]
    x_data={'transfer_values_input': transfer_values,'decoder_input': decoder_input_data}
    decoder_output = decoder_model.predict(x_data)
#    NB: non-normalised tokens are negative
#    DON'T NORMALISE TOKENS
#    otherwise 1.0 is always gonna win, as it's always the RELATIVE best
#   while we are interested in the absolute best
    token_onehot = decoder_output[0, count_tokens, :]
    first_choice=[np.argmax(token_onehot),np.max(token_onehot)]
    return first_choice


def get_test_captions(debug=0):
    test_captions=list()
    ctr=0
    for filename in filenames_test:
        print('Analysing ',filename)

        image_path=path+filename
        image=load_image(image_path,size=img_size)
        image_batch=np.expand_dims(image,axis=0)
        transfer_values=image_model_transfer.predict(image_batch)
        captions_list=beam_search(transfer_values)
        image_captions=list()
        for caption in captions_list:
            s=sequence_to_sentence(caption['sequence'])
            conf=getAvgConfidence(caption)
            cap={'sentence':s,'score':conf}
            image_captions.append(cap)
        test_captions.append(copy.copy(image_captions))
        
        
        ctr+=1
        progress=100*ctr/len(filenames_test)
        print('processed ',ctr,'images\t%.2f%%'% progress)
        if debug:
            if ctr==3:
                return test_captions
    return test_captions
    

tv_shape=transfer_values_test[0].shape[0]
def get_test_captions_tv(debug=0):
    test_captions=list()
    ctr=0
    for i in range(len(filenames_train)):
        transfer_values=transfer_values_test[i]
        transfer_values=np.reshape(transfer_values,(1,tv_shape))
        captions_list=beam_search(transfer_values)
        image_captions=list()
        for caption in captions_list:
            s=sequence_to_sentence(caption['sequence'])
            conf=getAvgConfidence(caption)
            cap={'sentence':s,'score':conf}
            image_captions.append(cap)
        test_captions.append(copy.copy(image_captions))
        
        
        ctr+=1
        print_progress(i+1,len(filenames_train))
        if debug:
            if ctr==3:
                return test_captions
    return test_captions

# This code normalises a vector in a [0,1] range
def normalise(vector):
    M=np.max(vector)
    m=np.min(vector)
    vector=(vector-m)/(M-m)
    return vector

def predict_next_word(transfer_values,prev_sequence,count_tokens,guessNr):
    x_data=\
    {
     'transfer_values_input':transfer_values,
     'decoder_input':prev_sequence
     }
    decoder_output=decoder_model.predict(x_data)
#    compute the softmax in order to get confidence scores between 0 and 1
    token_onehot = decoder_output[0, count_tokens, :]
    token_onehot = softmax(token_onehot)

    [outToken,confidence]=nth_best(token_onehot,guessNr)
    outWord=tokenizer.token_to_word(outToken)
    
    return outToken,confidence

nr_guesses=3
def get_guesses(transfer_values,caption,prev_confidence):
#    if the caption is already completed, don't make further guesses
    if token_end in caption:
        return caption
    new_captions=list()
    count_token=get_tokencount(caption)
    for guess_iter in range(1,nr_guesses+1):
        [token,confidence]=predict_next_word(transfer_values,caption,count_token,guess_iter)
        new_sequence=copy.copy(caption)
        new_sequence[0,count_token+1]=token
        new_caption={'sequence':copy.copy(new_sequence),'confidence':prev_confidence+confidence}
        new_captions.append(copy.copy(new_caption))
    return new_captions
def get_tokencount(sequence):
    ctr=-1
    for i in range(30):
        if sequence[0,i]==0:
            break
        ctr+=1
    return ctr
def sequence_to_sentence(sequence,verbose=0):
    length=sequence.shape[1]
    s=""
    for i in range(length):
        t=sequence[0,i]
        if t==0:
            break
        w=tokenizer.token_to_word(t)
        s+=" "
        s+=w
    if verbose:
        print(s)
    s=s.replace('ssss ','')
    s=s.replace(' eeee','')
    return s


sent_len=29
def beam_search(transfer_values):
    starter_sequence=np.zeros(shape=(1,30),dtype=np.int)
    starter_sequence[0,0]=token_start
    caption_list=list()
    starter_caption={
            'sequence':starter_sequence,
            'confidence':0
            }
    caption_list.append(starter_caption)
    for i in range(sent_len):
        new_captions=list()
        for caption in caption_list:
# if caption is complete, automatically save it
            if isComplete(caption):
                new_captions.append(copy.copy(caption))
                continue
            guesses=get_guesses(transfer_values,caption['sequence'],caption['confidence'])
            for guess in guesses:
                new_captions.append(copy.copy(guess))
#        caption_list=list()
#        caption_list=copy.copy(new_captions)
#        print(i)
#        only keep n of the best captions
        confVector=list()
        for caption in new_captions:
            conf=getAvgConfidence(caption)
            confVector.append(conf)
#        get the n best confidences
        guesses2keep=5
        best_guesses=list()
        for bestGuess_iter in range(1,min(guesses2keep+1,len(confVector)+1)):
            [best_position,best_value]=nth_best(confVector,bestGuess_iter)
#            debug print
#            print(best_guesses)
#            print(new_captions[best_position])
#            if new_captions[best_position] not in best_guesses:
            best_guesses.append(copy.copy(new_captions[best_position]))
        caption_list=list()
        caption_list=copy.copy(best_guesses)
#        if all the captions are complete, save the list and return it
        if allComplete(caption_list):
            return caption_list
#        print('candidate captions: ',len(caption_list))
    return caption_list
# This function returns the n-th highest value in a vector
def nth_best(vector,n):
    v=copy.copy(vector)
#    discard (n-1) biggest elements
    for i in range(n-1):
        best=np.argmax(v)
        v[best]=-100
    best_position=np.argmax(v)
    best_value=max(v)
    return best_position,best_value
# This code normalises a vector in a [0,1] range
def normalise(vector):
    M=np.max(vector)
    m=np.min(vector)
    vector=(vector-m)/(M-m)
    return vector


def getAvgConfidence(caption):
    sequence=caption['sequence']
    confidence=caption['confidence']
#    get the length of the sequence
    l=getSeqLen(sequence)
    avgConfidence=confidence/l
    return avgConfidence

# this function checks whether the current caption is complete
def isComplete(caption):
    if 2 in caption['sequence']:
        return True
    else: 
        return False
# this function checks if all the captions in the list are complete
def allComplete(capList):
    allComplete=True
    for c in capList:
        if isComplete(c):
            continue
#        this never gets executed if all the captions are complete
        allComplete=False
    return allComplete
def getSeqLen(sequence,verbose=0):
    l=0
    for i in range(30):
        if sequence[0,i]==0:
            break
        l+=1
        if sequence[0,i]==2:
            break
    if verbose:
        print(l)
    return l