# ========================================================================
# Copyright 2022 Emory University
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
import json
from collections import Counter
from typing import Dict, Any, List
import numpy as np

from src.vector_space_models import tf_idfs, most_similar

FM = {
    'antgrass2.ram': 'TheAntsandtheGrasshopper',
    'TheAssandhisPurchaser2': 'TheAssandHisPurchaser',
    'TheAssandtheLapdog2': 'TheAssandtheLapdog',
    'TheAssintheLionSkin': 'TheAssintheLionsSkin',
    'TheBellyandtheMembers2': 'TheBellyandtheMembers',
    'TheBuffoonandtheCountryman': 'TheBuffoonandtheCountryman2',
    'TheCrowandthePitcher2': 'crowpitc2.ram',
    'TheDogintheManger2': 'TheDogintheManger',
    'TheDogandtheShadow2': 'TheDogandtheShadow',
    'TheEagleandtheArrow2': 'TheEagleandtheArrow',
    'TheFoxandtheCrow2': 'TheFoxandtheCrow',
    'TheFoxandtheGoat2': 'TheFoxandtheGoat',
    'TheFoxandtheGrapes2': 'TheFoxandtheGrapes',
    'TheFoxandtheLion': 'TheFoxandtheLion2',
    'TheFoxandtheMask': 'foxmask2.ram',
    'haretort2.ram': 'TheHareandtheTortoise',
    'harefrog2.ram': 'TheHaresandtheFrogs',
    'TheHorseandtheAss2': 'TheHorseandtheAss',
    'TheLionandtheMouse2': 'lionmouse',
    'TheLioninLove2': 'TheLioninLove',
    'TheManandtheSatyr2': 'TheManandtheSatyr',
    'MercuryandtheWoodman': 'MercuryandtheWorkmen',
    'milkpail2.ram': 'milkmaidjug.jpg',
    'TheOldManandDeath2': 'TheOldManandDeath',
    'TheOldWomanandtheWine-Jar': 'womanjar2.ram',
    'TheOne-EyedDoe': 'TheOneEyedDoe',
    'ThePeacockandJuno': 'ThePeacockandJuno2',
    'TheRoseandtheAmaranth': 'TheRoseandtheAmaranth2',
    'TheSerpentandtheEagle': 'TheSerpentandtheEagle2',
    'shepherd2.ram': 'shepwolf2.ram',
    'TheSickLion2': 'TheSickLion',
    'TheTownMouseandtheCountryMouse': 'TheTownMouseandtheCountryMouse2',
    'TheTrumpeterTakenPrisoner2': 'TheTrumpeterTakenPrisoner',
    'TheTwoPots2': 'twopots2.ram',
    'TheVainJackdaw': 'TheVainJackdaw2',
    'TheWolfandtheCrane2': 'TheWolfandtheCrane',
    'TheWolfandtheLamb2': 'TheWolfandtheLamb',
    'TheWolfinSheepsClothing2': 'TheWolfinSheepsClothing'
}


def cosine(x1: Dict[str, float], x2: Dict[str, float]) -> float:
    x1, x2 = generate_sparse_vector(x1, x2)
    return np.inner(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def vectorize(documents: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    # Feel free to update this function
    return tf_idfs(documents)

def generate_sparse_vector(vec1, vec2): # stolen from stackexchange, makes it so the dimensions match
    counter1 = Counter(vec1)
    counter2 = Counter(vec2)
    all_items = set(counter1.keys()).union(set(counter2.keys()))
    vector1 = np.array([counter1[k] for k in all_items])
    vector2 = np.array([counter2[k] for k in all_items])
    return vector1, vector2


def similar_documents(X: Dict[str, Dict[str, float]], Y: Dict[str, Dict[str, float]]) -> Dict[str, str]:
    # similar documents will now use cosine similarity instead of euclidean distance
    # for each doc in x, find the most similar doc in y
    # return a dictionary mapping the name of the document in X to the name of the document in Y
    map = {}
    for doc in X:
        vec1 = X[doc]
        for doc2 in Y:
            vec2 = Y[doc2]
            if doc not in map:
                map[doc] = (doc2, cosine(vec1, vec2))
            else:
                if cosine(vec1, vec2) > map[doc][1]:
                    map[doc] = (doc2, cosine(vec1, vec2))
    # remove the scores
    for doc in map:
        map[doc] = map[doc][0]
    return map

# by changing it so similar documents used cosine similarity instead of euclidian distance
# and creating sparse vector mappings for all of the documents such that cosine similarity would work
# similar_documents is not more accurate. earlier, it could barely tell what the similar ones were.

if __name__ == '__main__':
    fables = json.load(open('../../res/vsm/aesopfables.json'))
    fables_alt = json.load(open('../../res/vsm/aesopfables-alt.json'))

    v_fables = vectorize(fables)
    v_fables_alt = vectorize(fables_alt)

    for x, y in similar_documents(v_fables_alt, v_fables).items():
        print('{} -> {}'.format(x, y))
