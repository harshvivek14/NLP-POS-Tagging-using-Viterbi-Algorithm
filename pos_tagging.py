
## POS tagging using Viterbi

# In this assignment, we need to implement the Viterbi algorithm for POS tagging

### Data Collection


import streamlit as st
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import pprint, time
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from numpy import array
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

import nltk
nltk.download('treebank')
nltk.download('universal_tagset')

from nltk.corpus import treebank

# we are using universal tagset as mentioned in the assignment requirements
data = list(nltk.corpus.treebank.tagged_sents(tagset='universal'))


# Getting list of train and test tagged words

def get_words(train_set, test_set):
    train_tagged_words = [tup for sent in train_set for tup in sent]
    test_tagged_words = [tup[0] for sent in test_set for tup in sent]
    train_tagged_tokens = [tag[0] for tag in train_tagged_words]
    train_tagged_pos_tokens = [tag[1] for tag in train_tagged_words]
    training_vocabulary_set = set(train_tagged_tokens)
    training_pos_tag_set = set(train_tagged_pos_tokens)

    return train_tagged_words, test_tagged_words, train_tagged_tokens, train_tagged_pos_tokens, training_vocabulary_set, training_pos_tag_set

# compute emission probability for a given word for a given tag
def word_given_tag(word, tag, train_bag):
    tag_list = [pair for pair in train_bag if pair[1] == tag]
    tag_count = len(tag_list)
    word_given_tag_list = [pair[0] for pair in tag_list if pair[0] == word]
    word_given_tag_count = len(word_given_tag_list)

    return (word_given_tag_count, tag_count)

# compute transition probabilities of a previous and next tag
def t2_given_t1(t2, t1, train_bag):
    tags = [pair[1] for pair in train_bag]

    t1_tags_list = [tag for tag in tags if tag == t1]
    t1_tags_count = len(t1_tags_list)

    t2_given_t1_list = [tags[index+1] for index in range(len(tags)-1) if tags[index] == t1 and tags[index+1] == t2]
    t2_given_t1_count = len(t2_given_t1_list)

    return(t2_given_t1_count, t1_tags_count)

# computing P(w/t) and storing in [Tags x Vocabulary] matrix.
# This is a matrix with dimension of len(training_pos_tag_set) X en(training_vocabulary_set)
def create_transition_matrix(training_pos_tag_set, training_vocabulary_set, train_tagged_words):
    len_pos_tags = len(training_pos_tag_set)
    len_vocab = len(training_vocabulary_set)

    # creating t x t transition matrix of training_pos_tag_set
    # each column is t2, each row is t1
    # thus M(i, j) represents P(tj given ti)

    tags_matrix = np.zeros((len_pos_tags, len_pos_tags), dtype='float32')
    for i, t1 in enumerate(list(training_pos_tag_set)):
        for j, t2 in enumerate(list(training_pos_tag_set)):
            tags_matrix[i, j] = t2_given_t1(t2, t1, train_tagged_words)[0]/t2_given_t1(t2, t1, train_tagged_words)[1]

    # convert the matrix to a df for better readability
    tags_df = pd.DataFrame(tags_matrix, columns = list(training_pos_tag_set), index=list(training_pos_tag_set))
    return tags_df

# Viterbi Algorithm

# The steps are as follows:

# 1. Given a sequence of words.
# 2. iterate through the sequence
# 3. for each word (starting from first word in sequence) calculate the product of emission probabilties and transition probabilties for all possible tags.
# 4. assign the tag which has maximum probability obtained in step 3 above.
# 5. move to the next word in sequence to repeat steps 3 and 4 above.


def Viterbi(words, train_tagged_words, tags_df):
    state = []
    T = list(set([pair[1] for pair in train_tagged_words]))

    for key, word in enumerate(words):
        #initialise list of probability column for a given observation
        p = []
        p_transition =[] # list for storing transition probabilities
        for tag in T:
            if key == 0:
                transition_p = tags_df.loc['.', tag]
            else:
                transition_p = tags_df.loc[state[-1], tag]

            # compute emission and state probabilities
            emission_p = word_given_tag(words[key], tag, train_tagged_words)[0]/word_given_tag(words[key], tag, train_tagged_words)[1]
            state_probability = emission_p * transition_p
            p.append(state_probability)
            p_transition.append(transition_p)

        pmax = max(p)
        state_max = T[p.index(pmax)]


        # if probability is zero (unknown word) then use transition probability
        if(pmax==0):
            pmax = max(p_transition)
            state_max = T[p_transition.index(pmax)]

        else:
            state_max = T[p.index(pmax)]

        state.append(state_max)
    return list(zip(words, state))

def confusion_matrix_creation(test_run_base, tagged_seq):
    test_data_tags = []
    for i in test_run_base:
        test_data_tags.append(i[1])
    pred_tags = []
    for i in tagged_seq:
        pred_tags.append(i[1])
    confusion_matrix = metrics.confusion_matrix(test_data_tags, pred_tags)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = list(set(pred_tags).union((set(test_data_tags)))))
    cm_display.plot()
    plt.xticks(rotation = 90)
    plt.show()

def testing(test_set):
    # choose random 20 sents only because it takes huge amount of time for evaluation
    random.seed(1234)
    rndom = [random.randint(1, len(test_set)) for x in range(15)]
    test_run = [test_set[i] for i in rndom]
    test_run_base = [tup for sent in test_run for tup in sent]
    test_tagged_words = [tup[0] for sent in test_run for tup in sent]

    # tagging the test sentences
    tagged_seq = Viterbi(test_tagged_words, train_tagged_words, create_transition_matrix(training_pos_tag_set, training_vocabulary_set, train_tagged_words))

    # accuracy
    viterbi_word_check = [i for i, j in zip(tagged_seq, test_run_base) if i == j]
    viterbi_accuracy = len(viterbi_word_check)/len(tagged_seq) * 100
    print('Viterbi Algorithm Accuracy: ', viterbi_accuracy)
    print('\n')
    return test_run_base, tagged_seq

def accuracy_per_tag(y_act, y_pred):
    d = {}
    for i in range(len(y_act)):
      if y_act[i] in d:
        if y_act[i] == y_pred[i]:
          d[y_act[i]] += [1]
        else:
          d[y_act[i]] += [0]
      else:
        if y_act[i] == y_pred[i]:
          d[y_act[i]] = [1]
        else:
          d[y_act[i]] = [0]

    accuracy_dict = {}
    for i in d:
      accuracy_dict[i] = sum(d[i])/len(d[i])
    return accuracy_dict


def predict_POS(sent):
  words = []
  train_tagged_words, test_tagged_words, train_tagged_tokens, train_tagged_pos_tokens, training_vocabulary_set, training_pos_tag_set = get_words(data, [])
  words.append(Viterbi(sent.split(), train_tagged_words, create_transition_matrix(training_pos_tag_set, training_vocabulary_set, train_tagged_words)))
  return(words)

predict_POS('You are not so looking well')

st.title("POS Tagging using Viterbi Algorithm") 
input = st.text_input("Enter the sentence 👇🏻") 
output = predict_POS(input)
if input:
    st.write("POS tagging", output[0])


