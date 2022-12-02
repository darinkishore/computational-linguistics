# ========================================================================
# Copyright 2021 Emory University
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

import re

RE_TOK = re.compile(r'([",.]|n\'t|\s+)')
NUMBERS = {'one': 1, 'two': 2,
           'three': 3, 'four': 4, 'five': 5, 'six': 6,
           'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11,
           'twelve': 12, 'thirteen': 13, 'fourteen': 14,
           'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
           'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
           'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90}

PLACES = {'hundred': 100, 'thousand': 1000, 'million': 1000000,
           'billion': 1000000000, 'trillion': 1000000000000}


def tokenize_regex(text):
    prev_idx = 0
    tokens = []
    for m in RE_TOK.finditer(text):
        t = text[prev_idx:m.start()].strip()
        if t: tokens.append(t)
        t = m.group().strip()
        if t:
            if tokens and tokens[-1] in {'Mr', 'Ms'} and t == '.':
                tokens[-1] = tokens[-1] + t
            else:
                tokens.append(t)
        prev_idx = m.end()

    t = text[prev_idx:]
    if t: tokens.append(t)
    return tokens

def is_num(string):
    return string.lower() in NUMBERS or string in PLACES

def is_part_of_fraction_decimal_ordinal(text:list, token, index):
    """
    looks at the token and sees if it is a part of or surrounding a blacklisted class
    first, fifth).
    five point two).
    a half, two thirds
    """

    blacklist = {'point', 'half', 'first', 'second' 'third',
                 'fourth' 'quarter', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth',
                 'eleventh', 'twelfth', 'thirteenth', 'fourteenth', 'fifteenth', 'sixteenth',
                 'seventeenth', 'eighteenth', 'nineteenth', 'twentieth', 'thirtieth', 'fortieth',
                 'fiftieth', 'sixtieth', 'seventieth', 'eightieth', 'ninetieth', 'hundredth',
                 'thousandth', 'millionth', 'billionth', 'trillionth',
                 'thirds', 'fourths', 'quarters', 'fifths', 'sixths', 'sevenths', 'eighths',
                 'ninths', 'tenths', 'elevenths', 'twelfths', 'thirteenths', 'fourteenths',
                 'fifteenths', 'sixteenths', 'seventeenths', 'eighteenths', 'nineteenths',
                 'twentieths', 'thirtieths', 'fortieths', 'fiftieths', 'sixtieths', 'seventieths',
                 'eightieths', 'ninetieths', 'hundredths', 'thousandths', 'millionths'}

    # if the token is a number, see if any part of the next or previous token without an s is in the blacklist
    if is_num(token):
        if index + 1 < len(text) and text[index + 1].lower() in blacklist \
                or text[index - 1].lower() in blacklist:
            return True

    if token.lower() in blacklist:
        return True

    return False

def get_sequences(text: list):
    """
    gets all the number sequences in the text
    """
    sequences = []
    for i, token in enumerate(text):
        # if this token is a number
        if is_num(token) and not is_part_of_fraction_decimal_ordinal(text, token, i):
            # if the previous token was a number
            if i > 0 and is_num(text[i - 1]):
                # if the previous sequence is not complete
                if sequences[-1][1] is None:
                    # if the next token is a number, dont complete the sequence
                    if i < len(text) - 1 and is_num(text[i + 1]):
                        continue
                    else:
                        sequences[-1][1] = i
                # if the previous sequence is complete
                else:
                    # if the next token is not a number
                    if i + 1 < len(text) and not is_num(text[i + 1]):  # note the added check for list index
                        sequences.append([i, i])
                    else:
                        sequences.append([i, None])
            # if the previous token was not a number
            else:
                # if the next token is not a number
                if i + 1 < len(text) and not is_num(text[i + 1]):
                    sequences.append([i, i])
                else:
                    sequences.append([i, None])

    # turn sequences into tuples
    sequences = [tuple(sequence) for sequence in sequences]
    return sequences

def get_numbers_from_sequence(sequence_indices, text: list):
    """
    Given a sequence of indices, return the number
    :param sequence_indices:
    :param text:
    :return dict:
    """
    sequence_numbers = {}

    for sequence in sequence_indices:
        sequence_numbers[sequence] = 0

    for start, end in sequence_indices:
        number = 0
        i = start
        while i < end + 1:
            token = text[i]
            if i + 1 < len(text) and text[i + 1] in PLACES:
                number += NUMBERS[token.lower()] * PLACES[text[i + 1].lower()]
                i += 1  # skip the place
            elif token.lower() in NUMBERS:
                number += NUMBERS[token.lower()]
            else:
                number += PLACES[token.lower()]
            i += 1
        sequence_numbers[(start, end)] = number
    return sequence_numbers

def find_and_replace(text, text_list, sequences, sequence_numbers):
    """
    Given a string and a number, replace all occurrences
    of that (whole ass) number with the digit version.
    """

    for sequence in sequences:
        starting_number = text_list[sequence[0]]
        ending_number = text_list[sequence[1]]

        if starting_number == ending_number:
            pattern = re.compile(rf'{starting_number}')
        else:
            pattern = re.compile(rf'({starting_number}.+?{ending_number})')

        matches = re.findall(pattern, text)
        for match in matches:
            text = text.replace(match, str(sequence_numbers[sequence]))

    # look for any indefinite articles in front of a digit number and remove them
    text = re.sub(r'\b(a|an)\s(\d+)', r'\2', text)
    return text


def normalize(text):
    """
    Turns cardinal numbers in a string to digits
    """
    text_string = text
    text_list = tokenize_regex(text)
    print(text)
    sequences = get_sequences(text_list)
    sequence_numbers = get_numbers_from_sequence(sequences, text_list)    # dict of sequence indices to number

    text = find_and_replace(text_string, text_list, sequences, sequence_numbers)

    return text


def normalize_extra(text):
    # TODO: to be updated
    return text


if __name__ == '__main__':

    S = [
        'I met twelve people',
        'I have one brother and two sisters',
        'A year has three hundred sixty five days',
        'I made a million dollars',
        "This's that ice-cold, twenty five, that white @@@ gold, this five hundred eighty for them",
        "five point two",
        "three fourths"
    ]

    T = [
        'I met 12 people',
        'I have 1 brother and 2 sisters',
        'A year has 365 days',
        'I made 1000000 dollars',
        "This's that ice-cold, 25, that white @@@ gold, this 580 for them",
        "five point two",
        "three fourths"
    ]

    correct = 0
    for s, t in zip(S, T):
        if normalize(s) == t:
            correct += 1

    print('Score: {}/{}'.format(correct, len(S)))
