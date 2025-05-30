import ast
import sys

from data import load, load_all, save
from config import cleaned_en_path, identify_dir, translate_dir
from utils import get_response
# from embeddings import read_faiss_index, retrieve_similar_words, load_embedding_matrix

import pandas as pd
pd.options.mode.chained_assignment = None


def identify_pun_meanings(df, model, start=0, end=-1):
  def apply(row):
    text_clean = row['text_clean']
    schema = '{ "pun_word": "pun_word", "pun_type": "pun_type", "first_meaning": [list of synonyms], "second_meaning": [list of synonyms], "first_context": [list of context words], "second_context": [list of context words] }'
    prompt = f"""
      Text: {text_clean}
    
      Step 1: Identify the pun word in this text. Output one word.
      Step 2: Does the pun play on root words that are spelled the same (homographic) or does the pun play on root words that are spelled differently but sound the same (homophonic). Output either the word "homographic" or the word "homophonic".
      Step 3: Make a list of synonyms for each of the two meanings of the pun. Output two lists: one list of synonyms for the first meaning of the pun and another list of synonyms for the second meaning of the pun. If it is a homophonic pun include the homophones in the appropriate lists.
      Step 4: For each of the two meanings, identify any context words in the text that clearly support the respective meaning. Do not include context words unless they clearly support the meaning.
      
      Return the output of the steps as a properly formatted json using this schema: {schema}
    """

    print(row.name, text_clean)
    try:
      response = get_response(prompt, model)
    except ValueError as e:
      print(f'Error: {e}')
      response = '{ "pun_word": "ERROR", "pun_type": "", "first_meaning": [], "second_meaning": [], "first_context": [], "second_context": [] }'
      pass
    return response

  chunk_size = 100
  chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
  if end == -1:
    end = len(chunks)
  for i in range(start, end):
    chunks[i][['pun_word', 'pun_type', 'first_meaning', 'second_meaning', 'first_context', 'second_context']] = chunks[i].apply(apply, axis=1)
    save(chunks[i], f'{identify_dir}{model}/{i}.tsv')


def translate_pun_meanings(df, model, start=0, end=-1):
  def apply(row):
    r = row.to_dict()
    pun_word = r['pun_word']
    first_meaning = r['first_meaning'].replace("'", '"')
    second_meaning = r['second_meaning'].replace("'", '"')
    first_context = r['first_context'].replace("'", '"')
    second_context = r['second_context'].replace("'", '"')

    prompt = f"""
      For each item in the values in this json, translate the item into French. Do not change the keys. The output must be a correctly formatted json.
      {{ "pun_word_fr": "{pun_word}", "first_meaning_fr": {first_meaning}, "second_meaning_fr": {second_meaning}, "first_context_fr": {first_context}, "second_context_fr": {second_context} }}
    """
    print(row.name, pun_word, first_meaning, second_meaning)
    try:
      response = get_response(prompt, model)
    except ValueError as e:
      print(f'Error: {e}')
      response = '{ "pun_word": "ERROR", "pun_type": "", "first_meaning": [], "second_meaning": [], "first_context": [], "second_context": [] }'
      pass
    return response

  chunk_size = 100
  chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
  if end == -1:
    end = len(chunks)
  for i in range(start, end):
    chunks[i][['pun_word_fr', 'first_meaning_fr', 'second_meaning_fr', 'first_context_fr', 'second_context_fr']] = chunks[i].apply(apply, axis=1)
    save(chunks[i], f'{translate_dir}{model}/{i}.tsv')


# def find_phonetically_similar_matches(df):
#     index = read_faiss_index()
#     embedding_matrix = load_embedding_matrix()
#
#     def apply(row):
#       alternative_word_fr = row['alternative_word_fr']
#       return retrieve_similar_words(index, embedding_matrix, query_word=alternative_word_fr, top_k=5)
#
#     df['similar_words'] = df.apply(apply, axis=1)
#     return df


def generate_french_puns(df):
  # TODO: generate French puns using the information generated in previous steps
  return True
  # def prompt_llm(row):
  #   text_clean = row['text_clean']
  #   pun_type = row['pun_type']
  #   pun_word_fr = row['pun_word_fr']
  #   alternative_meaning_fr = row['alternative_meaning_fr']
  #
  #   schema = '{"is_homonym": 1 or 0, "meanings_overlap": 1 or 0}'
  #
  #   prompt = f"""
  #     Question 1: Is the French word "{pun_word_fr}" a homonym? If yes, output 1, else output 0.
  #     Question 2: Does the semantic range of the word "{pun_word_fr}" overlap with the semantic range of the word "{alternative_meaning_fr}"? If yes, output 1, else output 0.
  #
  #     Return the output of the steps as a json in this format (Do not include any additional text): {schema}
  #   """
  #   response = llm.invoke(prompt).content
  #   print(row.name, response[response.find('{'):response.rfind('}') + 1].replace('\n', ''))
  #   response_json = json.loads(response[response.find('{'):response.rfind('}') + 1])
  #
  #   if response_json['is_homonym'] == 1 and response_json['meanings_overlap'] == 1:
  #
  #
  #   is_pun = int(llm.invoke(prompt).content)
  #   pun = non_pun
  #   print(row.name, is_pun, non_pun)
  #
  #
  #
  #   schema = '{ "generated_french_pun": "generated_french_pun" }'
  #   prompt = f"""
  #     Step 1: Identify the pun word in this text. Output one word.
  #     Step 2: Is this is a homographic or homophonic pun. Output either the word "homophonic" or the word "homographic".
  #     Step 3: If the pun is homographic, identify the alternative meaning of the pun word that makes the text funny. Output one word.
  #     Step 4: If the pun is homophonic, identify the alternative word that the pun word alludes to. Output one word.
  #     Step 5: Pun words often occur within idiomatic phrases that support the alternative meaning. Identify the idiomatic phrase that makes the pun funny. Output a short phrase.
  #
  #     Return the output of the steps as a json in this format (Do not include any additional text): {schema}
  #   """
  #
  #   response = llm.invoke(prompt).content
  #   print(row.name, response[response.find('{'):response.rfind('}') + 1].replace('\n', ''))
  #   response_json = json.loads(response[response.find('{'):response.rfind('}') + 1])
  #   return pd.Series(response_json)
  #
  # df[['pun_word_en', 'pun_type', 'alternative_meaning_en', 'alternative_word_en', 'phrase_en']] = df.apply(prompt_llm,
  #                                                                                                          axis=1)
  # return df


if __name__ == "__main__":
  task = sys.argv[1]
  model = sys.argv[2]
  start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
  end = int(sys.argv[4]) if len(sys.argv) > 4 else -1

  if task == 'identify':
    df = load(cleaned_en_path)
    identify_pun_meanings(df, model, start, end)

  if task == 'translate':
    df = load(f'{identify_dir}{model}.tsv')
    translate_pun_meanings(df, model, start, end)


    # for i in range(len(df)):
    #   json_str = df[['pun_word', 'first_meaning', 'second_meaning', 'first_context', 'second_context']].iloc[i]
    #   print(json_str)

  # find_synonyms()
  # translate_pun_meanings()
  # find_phonetically_similar_matches()
  # generate_french_puns()


