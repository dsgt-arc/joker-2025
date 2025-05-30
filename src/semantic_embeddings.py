import fasttext.util
import scipy
from config import fasttext_en_path, fasttext_fr_path
import numpy as np

# fasttext.util.download_model('en', if_exists='ignore')  # English
# fasttext.util.download_model('fr', if_exists='ignore')  # French
fasttext_en = fasttext.load_model(fasttext_en_path)
fasttext_fr = fasttext.load_model(fasttext_fr_path)

def get_similarity_score(rows):
  similarities = []
  for row in rows:
    pun_word = get_similarity(row['pun_word'], row['pun_word_fr'])
    if pun_word and pun_word[1]:
      similarities.append(pun_word[0])

    pun_type = get_similarity(row['pun_type'], row['pun_type_fr'])
    if pun_type and pun_type[1]:
      similarities.append(pun_type[0])

    for i, word in enumerate(row['first_meaning']):
      first_meaning = get_similarity(row['first_meaning'][i], row['first_meaning_fr'][i])
      if first_meaning and first_meaning[1]:
        similarities.append(first_meaning[0])

    for i, word in enumerate(row['second_meaning']):
      second_meaning = get_similarity(row['second_meaning'][i], row['second_meaning_fr'][i])
      if second_meaning and second_meaning[1]:
        similarities.append(second_meaning[0])

    for i, word in enumerate(row['first_context']):
      first_context = get_similarity(row['first_context'][i], row['first_context_fr'][i])
      if first_context and first_context[1]:
        similarities.append(first_context[0])

    for i, word in enumerate(row['second_context']):
      second_context = get_similarity(row['second_context'][i], row['second_context_fr'][i])
      if second_context and second_context[1]:
        similarities.append(second_context[0])

  return np.mean(similarities, axis=0)


def get_similarity(text_en, text_fr):
  embbedding_en = fasttext_en["beam"]
  embbedding_fr = fasttext_fr["poutre"]

  found = not all(x == 0 for x in embbedding_fr)

  return scipy.spatial.distance.cosine(embbedding_en, embbedding_fr), found

if __name__ == "__main__":
  rows = [{  "pun_word": "yolk",  "pun_type": "homophonic",  "first_meaning": ["vitellus", "yolk"],  "second_meaning": ["harness", "beam", "crosspiece", "coupling", "yoke"],  "first_context": ["egg"],  "second_context": ["pulls", "cart"],
    "pun_word_fr": "jaune d'œuf",
    "pun_type_fr": "homophonique",
    "first_meaning_fr": ["vitellus", "jaune d'œuf"],
    "second_meaning_fr": ["harnais", "poutre", "traverse", "attelage", "joug"],
    "first_context_fr": ["œuf"],
    "second_context_fr": ["tire", "charrette"]
  }]

  print(get_similarity_score(rows))


