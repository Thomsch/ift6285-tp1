#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %%
from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
import gzip

batch_size = 128  # Batch size for training.
epochs = 2  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
# Path to the data txt file on disk.

import os

os.chdir("/Users/louis/Google Drive/M.Sc-DIRO-UdeM/IFT6285-Traitements automatique des langues naturelles/ift6285-tp1")
# os.chdir("/Users/fanxiao/Google Drive/UdeM/IFT6135 Representation Learning/homework1/programming part ")
print(os.getcwd())

# %%
# Vectorize the data.
input_characters = set()
target_characters = set()


def loadData(corpuspath, size=None):
    input_texts = []
    target_texts = []
    with gzip.open(corpuspath, 'rt', encoding='ISO-8859-1') as f:
        lines = f.read().split('\n')
    input_phrase = []
    target_phrase = []
    # for line in lines[: min(num_samples, len(lines) - 1)]:
    i = 0
    for line in lines:
        if not line.startswith('#begin') and not line.startswith('#end') and len(line.split('\t')) > 1:
            line = line.encode("ascii", errors="ignore").decode()
            target_word, input_word = line.split('\t')
            input_word = input_word.strip().lower()
            target_word = target_word.strip().lower()
            input_phrase.append(input_word)
            target_phrase.append(target_word)
            input_phrase.append(' ')
            target_phrase.append(' ')
            if input_word == '.':
                # We use "tab" as the "start sequence" character
                # for the targets, and "\n" as "end sequence" character.
                input_texts.append(''.join(input_phrase))
                target_texts.append('\t' + ''.join(target_phrase) + '\n')
                input_phrase = []
                target_phrase = []

            for char in input_word:
                if char not in input_characters:
                    input_characters.add(char)
            for char in target_word:
                if char not in target_characters:
                    target_characters.add(char)
            i += 1
            if size is not None and i > size:
                break

    return input_texts, target_texts


input_texts, target_texts = loadData('data/train-1544.gz', 5000)
size_train = len(input_texts)
test_input_texts, test_target_texts = loadData('data/test-2834.gz', 2000)
size_test = len(test_input_texts)

input_texts = np.array(input_texts)
target_texts = np.array(target_texts)
# np.random.seed(456)
# indexs = np.arange(0,len(input_texts))
# np.random.shuffle(indexs)
# input_texts = input_texts[indexs[0:size_train]]
# target_texts=target_texts[indexs[0:size_train]]

test_input_texts = np.array(test_input_texts)
test_target_texts = np.array(test_target_texts)

# input_texts,target_texts: list of phrases
# input_token_index: key-values of char
# encoder_input_data:3dimension, d1=phrase,d2=char index in phrase,d3=index of list of candidate chars
print(input_texts[0:2])
print(target_texts[0:2])

# %%
input_characters.add(' ')
target_characters.add(' ')
target_characters.add('\t')
target_characters.add('\n')
input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of sentence samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)
print('sentence of training data:', size_train)
print('sentence of test data:', size_test)

input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

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

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.

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
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# %%
# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
# Save model
# model.save('s2s.h5')

print(os.getcwd())
from keras.models import load_model

model.save('model1-lstm-4016samples-100epochs.h5')
# del model  # deletes the existing model

# Save model
# from keras.models import load_model
# model.save('model1-lstm-4016samples-100epochs.h5')
# model.save('model1-lstm-4016samples-100epochs.h5')
# model = load_model('my_model.h5')

# model.save_weights('my_model_weights.h5')
# model.load_weights('my_model_weights.h5')


# %%
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
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


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
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
                len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


'''
Functions of Evalutate the prediction
'''


def count_accuracy(pred_sents, target_sents):
    count_accu = 0
    total = 0
    for pred_sent, target_sent in zip(pred_sents, target_sents):
        pred_list = pred_sent.split(' ')
        targ_list = target_sent[2:].split(' ')
        for pred_token, target_token in zip(pred_list, targ_list):
            total += 1
            if pred_token == target_token:
                count_accu += 1
    return count_accu, total


# %%
# print(os.getcwd())
# from keras.models import load_model
# model=load_model('output-lstm/model1-lstm-4016samples-100epochs.h5')


# %%
print('-------predict train data:')
train_pred_sents = []
for seq_index in range(500):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-- No. ', seq_index)
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)
    train_pred_sents.append(decoded_sentence)

train_acc_count, count_total = count_accuracy(train_pred_sents, input_texts)
print('Accuracy on LSTM1 predicteur, train data:', train_acc_count, '/', count_total, '=',
      train_acc_count / count_total)

# %%
print('-----predict test data:')

test_encoder_input_data = np.zeros(
    (len(test_input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
test_decoder_input_data = np.zeros(
    (len(test_input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
test_decoder_target_data = np.zeros(
    (len(test_input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

for i, (test_input_text, test_target_text) in enumerate(zip(test_input_texts, test_target_texts)):
    for t, char in enumerate(test_input_text):
        test_encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(test_target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        test_decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            test_decoder_target_data[i, t - 1, target_token_index[char]] = 1.

test_pred_sents = []
for seq_index in range(500):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = test_encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-- No. ', seq_index)
    print('Input test lemm sentence:', test_input_texts[seq_index])
    print('Input test surf sentence:', test_input_texts[seq_index])
    print('Prediction test sentence:', decoded_sentence)
    test_pred_sents.append(decoded_sentence)

test_acc_count, count_total = count_accuracy(test_pred_sents, test_target_texts)
print('Accuracy on LSTM1 predicteur, test data:', test_acc_count, '/', count_total, '=', test_acc_count / count_total)
