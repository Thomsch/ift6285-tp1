#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
import gzip


batch_size = 128  # Batch size for training.
epochs = 20  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
# Path to the data txt file on disk.

import os
os.chdir("/Users/louis/Google Drive/M.Sc-DIRO-UdeM/IFT6285-Traitements automatique des langues naturelles/ift6285-tp1")
# os.chdir("/Users/fanxiao/Google Drive/UdeM/IFT6135 Representation Learning/homework1/programming part ")
print(os.getcwd())


#%%
# Vectorize the data.
input_texts = []
target_texts = []
input_types = set()
target_types = set()

def loadData(corpuspath):
    with gzip.open(corpuspath, 'rt', encoding='ISO-8859-1') as f:
        lines = f.read().split('\n')
    input_phrase=[]
    target_phrase=[]
    # for line in lines[: min(num_samples, len(lines) - 1)]:
    for line in lines:
        if line.startswith('#begin') or line.startswith('#end'):
            continue
        line=line.encode("ascii", errors="ignore").decode()
        if  len(line.split('\t'))==2:
            target_word, input_word = line.split('\t')
            input_word=input_word.lower().strip()
            target_word=target_word.lower().strip()
            if input_word.startswith("'") and not input_word.startswith("''"):
                input_word=input_word[1:]
            if target_word.startswith("'") and not target_word.startswith("''"):
                target_word=target_word[1:]
            if input_word=='':
                continue
            input_phrase.append(input_word)
            target_phrase.append(target_word)
            if input_word not in input_types:
                input_types.add(input_word)
            if target_word not in target_types:
                target_types.add(target_word)
            # input_phrase.append(' ')
            # target_phrase.append(' ')
            if input_word=='.':
                # We use "tab" as the "start sequence" character
                # for the targets, and "\n" as "end sequence" character.
                input_texts.append(input_phrase)
                target_phrase.append('\n')
                target_texts.append(target_phrase)
                input_phrase=[]
                target_phrase=['\t']

    
loadData('data/train-1544.gz')
size_train=len(input_texts)
loadData('data/test-2834.gz')
size_test=len(input_texts)-size_train


input_texts=np.array(input_texts)
target_texts=np.array(target_texts)
np.random.seed(457)
indexs = np.arange(0,len(input_texts))
np.random.shuffle(indexs)
input_texts = input_texts[indexs[0:len(input_texts)]]
target_texts=target_texts[indexs[0:len(input_texts)]]

#input_texts,target_texts: list of phrases
#input_token_index: key-values of char
#encoder_input_data:3dimension, d1=phrase,d2=char index in phrase,d3=index of list of candidate chars
print(input_texts[0:2])
print(target_texts[0:2])

#%%
# input_characters.add(' ')
# target_characters.add(' ')
target_types.add('\t')
target_types.add('\n')
input_types = sorted(list(input_types))
target_types = sorted(list(target_types))
num_encoder_tokens = len(input_types)
num_decoder_tokens = len(target_types)

max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)
print('size of training data:', size_train)
print('size of test data:', size_test)

input_token_index = dict(
    [(typex, i) for i, typex in enumerate(input_types)])
target_token_index = dict(
    [(typey, i) for i, typey in enumerate(target_types)])

# print(input_token_index)
# print(target_token_index)

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
#%%
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, typex in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[typex]] = 1.
    for t, typey in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[typey]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[typey]] = 1.

all_input_texts=input_texts
input_texts=all_input_texts[0:size_train]
all_target_texts=target_texts
target_texts=all_target_texts[0:size_train]
all_encoder_input_data=encoder_input_data
encoder_input_data=all_encoder_input_data[0:size_train]
all_decoder_input_data=decoder_input_data
decoder_input_data=all_decoder_input_data[0:size_train]
all_decoder_target_data=decoder_target_data
decoder_target_data=all_decoder_target_data[0:size_train]

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


#%%
# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)

# Save model
# from keras.models import load_model
# model.save('model1-lstm-4016samples-100epochs.h5')
# model.save('model1-lstm-4016samples-100epochs.h5')
# model = load_model('my_model.h5')

# model.save_weights('my_model_weights.h5')
# model.load_weights('my_model_weights.h5')


#%%
# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_types_index = dict(
    (i, typex) for typex, i in input_token_index.items())
reverse_target_types_index = dict(
    (i, typey) for typey, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_type = reverse_target_types_index[sampled_token_index]
        decoded_sentence.append( sampled_type)

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_type == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence

#%%
print('-------predict train data:')
for seq_index in range(20):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)

print('-----predict test data:')
for seq_index in range(20):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = all_encoder_input_data[size_train+seq_index: size_train+seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input test sentence:', all_input_texts[size_train+seq_index])
    print('Decoded test sentence:', decoded_sentence)
