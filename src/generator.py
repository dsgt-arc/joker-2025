import sys

from data import load, load_all, save
from config import combined_en_path, identify_path
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

    ##Step 4: Pun words often occur within idiomatic phrases that support the alternative meaning. Identify the idiomatic phrase that makes the pun funny. Output a short phrase.

    print(row.name, text_clean)
    try:
      response = get_response(prompt, model)
    except ValueError:
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
    save(chunks[i], f'{identify_path}{model}/{i}.tsv')


def translate_pun_meanings(df):
  # TODO: translate all of the synonyms, not just the two meanings
  def apply(row):
    pun_word = row['pun_word_en']
    alternative_word = row['alternative_word_en']
    idiomatic_phrase = row['phrase_en']
    schema = '{"pun_word": "step_1_word", "alternative_word": "step_2_word", "idiomatic_phrase": "step_3_phrase"}'

    prompt = f"""
      Step 1: Translate the word "{pun_word}" into French. Output one word.
      Step 2: Translate the word "{alternative_word}" into French. Output one word.
      Step 3: Translate this idiomatic phrase into French: {idiomatic_phrase}. Do not translate literally. Output a short French idiomatic phrase.
      
      Return the output of the steps as a json using this schema: {schema}
    """
    print(row.name, pun_word, alternative_word, idiomatic_phrase)
    return get_response(prompt, model)

  df[['pun_word_fr', 'alternative_word_fr', 'phrase_fr']] = df.apply(apply, axis=1)
  return df


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
  model = sys.argv[1]

  ### identify pun meanings
  # start = int(sys.argv[2])
  # end = int(sys.argv[3])
  # df = load(combined_en_path)#.head(5)
  # identify_pun_meanings(df, model, start, end)

  ### combine all identify batch files
  # model_df = load_all(f'{identify_path}{model}/')
  # save(model_df, f'{identify_path}{model}.tsv')

  # find_synonyms()
  # translate_pun_meanings()
  # find_phonetically_similar_matches()
  # generate_french_puns()


