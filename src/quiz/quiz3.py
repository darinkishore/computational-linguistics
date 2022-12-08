# ========================================================================
# Copyright 2020 Emory University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========================================================================
import functools
import pickle
from collections import Counter
from typing import List, Tuple, Dict, Any
from memoization import cached
import numpy as np
import pandas as pd




### approach: use hidden markov model and viterbi algorithm to predict the most likely sequence of tags
### for a given sequence of words

### the model is trained on the training data and the parameters are stored in a pickle file

### steps:
### 1. read the training and development data
### 2. create transition and emission dictionaries
### 3. create a dictionary of dictionaries for the transition and emission probabilities

# based on https://towardsdatascience.com/hidden-markov-model-implemented-from-scratch-72865bda430e
# and Stanford's textbook 8.4.1
class HMM:
    def __init__(self, training_data: List[List[Tuple[str, str]]], dev_data: List[List[Tuple[str, str]]]):
        self.data = training_data
        self.dev_data = dev_data
        self.parts_of_speech = {'ADD', 'AFX', 'CC', 'CD', 'CODE', 'DT', 'EX', 'FW', 'GW', 'IN',
                  'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
                  'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'TO', 'UH', 'VB', 'VBD',
                  'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', 'XX', '$', ':', ',',
                  '.', '``', '\'\'', '#', '-LRB-', '-RRB-', 'HYPH', 'NFP', 'SYM', 'PUNC'}
        self.state_counts = {state: 0 for state in self.parts_of_speech}
        self.following_state_counts = {state: {state: 0 for state in self.parts_of_speech} for state in self.parts_of_speech}
        self.transition_matrix = self.calculate_transition_probabilities(self.data, self.following_state_counts, self.state_counts) # change to self.dev_data for dev data
        self.transition_matrix = self.create_transition_matrix()
        self.all_words = set()
        self.emission_matrix = self.create_emission_matrix(self.data)

        # self.emission_matrix = self.create_emission_matrix(

    @cached
    def calculate_transition_probabilities(self, data, following_state_counts, state_counts):
        for sentence in data:
            for i, word in enumerate(sentence):
                if i == 0:
                    state_counts[word[1]] += 1
                    # set period to be the end of the previous sentence
                    following_state_counts['.'][word[1]] += 1
                else:
                    self.following_state_counts[sentence[i-1][1]][word[1]] += 1
                    self.state_counts[word[1]] += 1
        for state in self.parts_of_speech:
            for following_state in self.parts_of_speech:
                if self.state_counts[state] == 0:
                    self.following_state_counts[state][following_state] = 0
                else:
                    following_state_counts[state][following_state] /= (state_counts[state])
                # make sure that total probability of all following states is 1
        return following_state_counts

    @cached
    def create_transition_matrix(self):
        transition_matrix = pd.DataFrame(self.transition_matrix)
        return transition_matrix

    @cached
    def create_emission_matrix(self, data):
        words = set()
        for sentence in data:
            for word in sentence:
                words.add(word[0])
                self.all_words.add(word[0])

        emission_dict = {state: {word: 0 for word in words} for state in self.parts_of_speech}

        num_pos_per_state = self.state_counts

        for sentence in data:
            for word in sentence:
                # how many times does this word appear with this part of speech
                emission_dict[word[1]][word[0]] += 1

        for state in self.parts_of_speech:
            for word in words:
                if num_pos_per_state[state] == 0:
                    emission_dict[state][word] = 0
                else:
                    emission_dict[state][word] /= num_pos_per_state[state]

        emission_matrix = pd.DataFrame(emission_dict)
        return emission_matrix




def read_data(filename: str):
    data, sentence = [], []
    fin = open(filename)

    for line in fin:
        l = line.split()
        if l:
            sentence.append((l[0], l[1]))
        else:
            data.append(sentence)
            sentence = []

    return data

@cached
def to_probs(model: Dict[Any, Counter]) -> Dict[str, List[Tuple[str, float]]]:
    probs = dict()
    for feature, counter in model.items():
        ts = counter.most_common()
        total = sum([count for _, count in ts])
        probs[feature] = [(label, count/total) for label, count in ts]
    return probs


def evaluate(data: List[List[Tuple[str, str]]], *args):
    total, correct = 0, 0
    for sentence in data:
        tokens, gold = tuple(zip(*sentence))
        pred = [t[0] for t in predict(tokens, args)]
        total += len(tokens)
        correct += len([1 for g, p in zip(gold, pred) if g == p])
    accuracy = 100.0 * correct / total
    return accuracy

def train(trn_data: List[List[Tuple[str, str]]], dev_data: List[List[Tuple[str, str]]]) -> Tuple:
    """
    :param trn_data: the training set
    :param dev_data: the development set
    :return: a tuple of all parameters necessary to perform part-of-speech tagging
    """
    hmm = HMM(trn_data, dev_data)
    return hmm.transition_matrix, hmm.emission_matrix, hmm.parts_of_speech, hmm.all_words

def predict(tokens: List[str], *args) -> List[Tuple[str, float]]:
    """
    :param tokens: a list of tokens.
    :param args: a variable number of arguments
    :return: a list of tuple where each tuple represents a pair of (POS, score) of the corresponding token.
    """
    transition_matrix, emission_matrix, parts_of_speech, words = args[0][0], args[0][1], args[0][2], args[0][3]
    # algorithm from wikipedia's python-ish pseudocode mixed with the stanford book

    # trellis dict with tokens
    trellis = {state: {token: 0 for token in tokens} for state in parts_of_speech}
    trellis = pd.DataFrame(trellis)
    backpointer = {state: {token: 0 for token in tokens} for state in parts_of_speech}
    backpointer = pd.DataFrame(backpointer)

    # initialize trellis
    for state in parts_of_speech:
        # if the word is not in the training data, it's a rare word so set it to the most likely tag
        # if tokens[0] not in words:
            # trellis.loc[state, tokens[0]] = transition_matrix.loc['.', state] * emission_matrix.loc[state, 'NN']
        # else:
        trellis.loc[state, tokens[0]] = transition_matrix.loc['.', state] * emission_matrix.loc[tokens[0], state]
        backpointer.loc[state, tokens[0]] = '.'


    for observation in range(1, len(tokens)):
        for state in parts_of_speech:
            # if the token is not in the emission matrix, it's a rare word, so set it to the most likely tag
            # if tokens[observation] not in words:
            #     trellis.loc[tokens[observation], state] = trellis.loc[tokens[observation-1], 'NNP'] * transition_matrix.loc['NNP', state]
            #     backpointer.loc[state, tokens[observation]] = 'NNP'
            # else:
            trellis.loc[tokens[observation], state] = max([trellis.loc[tokens[observation-1], prev_state] * transition_matrix.loc[prev_state, state] * emission_matrix.loc[tokens[observation], state] for prev_state in parts_of_speech])
            backpointer.loc[tokens[observation], state] = np.argmax([trellis.loc[tokens[observation-1], prev_state] * transition_matrix.loc[prev_state, state] * emission_matrix.loc[tokens[observation], state] for prev_state in parts_of_speech])



    # backtrace
    best_path = []
    best_path.append(max([state for state in parts_of_speech], key=lambda state: trellis.loc[tokens[-1], state]))
    for observation in range(len(tokens)-1, 0, -1):
        best_path.append(backpointer.loc[best_path[-1], tokens[observation]])
    best_path.reverse()
    return list(zip(best_path, [0 for _ in range(len(best_path))]))



if __name__ == '__main__':
    path = '../../'  # path to the cs329 directory
    trn_data = read_data(path + 'res/pos/wsj-pos.trn.gold.tsv')
    dev_data = read_data(path + 'res/pos/wsj-pos.dev.gold.tsv')
    model_path = path + 'src/quiz/quiz3.pkl'

    hmm = HMM(trn_data, dev_data)
    # save model
    args = train(trn_data, dev_data)
    pickle.dump(args, open(model_path, 'wb'))
    # load model
    args = pickle.load(open(model_path, 'rb'))
    print(evaluate(trn_data, *args))