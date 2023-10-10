## POS tagging using Viterbi Algorithm

### Data Collection

import streamlit as st
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize 

import nltk
nltk.download('punkt')
nltk.download('treebank')
nltk.download('universal_tagset')

from nltk.corpus import treebank

data = list(nltk.corpus.treebank.tagged_sents(tagset='universal'))


# Creating list of train and test tagged words

def get_words(train_set, test_set):

    train_tagged_words = []
    for sent in train_set:
        for tup in sent:
            train_tagged_words.append(tup)

    test_tagged_words = []
    for sent in test_set:
        for tup in sent:
            test_tagged_words.append(tup[0])

    train_tagged_tokens = []
    for tag in train_tagged_words:
        train_tagged_tokens.append(tag[0])

    train_tagged_pos_tokens = []
    for tag in train_tagged_words:
        train_tagged_pos_tokens.append(tag[1])

    training_vocabulary_set = set(train_tagged_tokens)
    training_pos_tag_set = set(train_tagged_pos_tokens)

    return train_tagged_words, test_tagged_words, train_tagged_tokens, train_tagged_pos_tokens, training_vocabulary_set, training_pos_tag_set

# Calculate emission probability for a given word for a given tag

def word_given_tag(word, tag, train_bag):

    tag_list = []
    for pair in train_bag:
        if pair[1] == tag:
            tag_list.append(pair)

    tag_count = len(tag_list)

    word_given_tag_list = []
    for pair in tag_list:
        if pair[0] == word:
            word_given_tag_list.append(pair[0])

    word_given_tag_count = len(word_given_tag_list)

    return (word_given_tag_count, tag_count)

# Calculate transition probabilities of a previous and next tag

def t2_given_t1(t2, t1, train_bag):

    tags = []
    for pair in train_bag:
        tags.append(pair[1])

    t1_tags_list = []
    for tag in tags:
        if tag == t1:
            t1_tags_list.append(tag)

    t1_tags_count = len(t1_tags_list)

    t2_given_t1_list = []
    for index in range(len(tags)-1):
        if tags[index] == t1 and tags[index+1] == t2:
            t2_given_t1_list.append(tags[index+1])

    t2_given_t1_count = len(t2_given_t1_list)

    return(t2_given_t1_count, t1_tags_count)

# computing P(w given t) and storing in [Tags x Vocabulary] matrix.

def create_transition_matrix(training_pos_tag_set, training_vocabulary_set, train_tagged_words):
    len_pos_tags = len(training_pos_tag_set)
    len_vocab = len(training_vocabulary_set)

    # creating t x t transition matrix of training_pos_tag_set
    matrix = np.zeros((len_pos_tags, len_pos_tags), dtype='float32')

    for i, t1 in enumerate(list(training_pos_tag_set)):
        for j, t2 in enumerate(list(training_pos_tag_set)):
            matrix[i, j] = t2_given_t1(t2, t1, train_tagged_words)[0]/t2_given_t1(t2, t1, train_tagged_words)[1]

    # convert to df
    df = pd.DataFrame(matrix, columns = list(training_pos_tag_set), index=list(training_pos_tag_set))

    return df


# Viterbi Algorithm

# The steps are as follows:

# 1. Given a sequence of words, iterate over the sequence.
# 2. for each word (starting from first word in sequence) calculate the product of emission probabilties and transition probabilties for all possible tags.
# 3. assign the tag which has maximum probability obtained in step 2 above.
# 4. move to the next word in sequence to repeat steps 2 and 3 above.


# we have handled unknown word problem as well with the help of transition probability

def Viterbi(words, train_tagged_words, tags_df):

    tags = []
    for pair in train_tagged_words:
        tags.append(pair[1])

    T = list(set(tags))
    state = []

    for key, word in enumerate(words):
        p_transition =[]
        p = []

        for tag in T:
            if key == 0:
                transition_p = tags_df.loc['.', tag]
            else:
                transition_p = tags_df.loc[state[-1], tag]

            # calculating emission and state probabilities
            e_probability = word_given_tag(words[key], tag, train_tagged_words)[0]/word_given_tag(words[key], tag, train_tagged_words)[1]
            s_probability = e_probability * transition_p
            p_transition.append(transition_p)
            p.append(s_probability)


        state_max = T[p.index(max(p))]
        pmax = max(p)

        # if probability is zero i.e unknown word, then use transition probability

        if(pmax==0):
            pmax = max(p_transition)
            state_max = T[p_transition.index(pmax)]

        else:
            state_max = T[p.index(pmax)]

        state.append(state_max)
    return list(zip(words, state)), tags_df



def predict_POS(sent):
    words = []
    train_tagged_words, test_tagged_words, train_tagged_tokens, train_tagged_pos_tokens, training_vocabulary_set, training_pos_tag_set = get_words(data, [])
    words.append(Viterbi(word_tokenize(sent), train_tagged_words, create_transition_matrix(training_pos_tag_set, training_vocabulary_set, train_tagged_words))[0])
    return words


st.title("POS Tagging using Viterbi Algorithm") 
st.write("Group: Harshvivek, Udhay, Chetan\n\n")
input = st.text_input("Enter the sentence 👇🏻\n") 

if input:
    st.write("Processing..\n\n") 
    output = predict_POS(input)
    st.write("\nPOS tagging: \n")
    for i in output[0]:
        st.write(str(i[0]) + ' - ' + str(i[1]))


