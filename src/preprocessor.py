import ast
import sys
import torch

from data import load, load_all, save
from config import cleaned_en_path, homonym_dir, identify_dir, translate_dir, similarity_dir
from utils import get_model, get_response
# from embeddings import read_faiss_index, retrieve_similar_words, load_embedding_matrix

from langchain_core.documents import Document
from langchain_google_community import GoogleTranslateTransformer
from sentence_transformers import SentenceTransformer, util

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


def translate_pun_meanings(df, model, start=0, end=-1, translate_flag=True):
  def translate(row):
    row_dict = row.to_dict()
    pun_word = row_dict['pun_word']
    first_meaning = row_dict['first_meaning'].replace("'", '"')
    second_meaning = row_dict['second_meaning'].replace("'", '"')
    first_context = row_dict['first_context'].replace("'", '"')
    second_context = row_dict['second_context'].replace("'", '"')

    prompt = f"""
      Translate the values in this json from English into French. If a value is a list, translate each element in the list. Do not change the keys. The output must be a correctly formatted json.
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

  def back_translate(row):
    r = row.to_dict()
    pun_word = r['pun_word_fr']
    first_meaning = r['first_meaning_fr']#.replace("'", '"')
    second_meaning = r['second_meaning_fr']#.replace("'", '"')
    first_context = r['first_context_fr']#.replace("'", '"')The semc
    second_context = r['second_context_fr']#.replace("'", '"')

    prompt = f"""
      Translate the values in this json from French into English. If a value is a list, translate each element in the list. Do not change the keys. The output must be a correctly formatted json.
      {{ "pun_word_bt": "{pun_word}", "first_meaning_bt": {first_meaning}, "second_meaning_bt": {second_meaning}, "first_context_bt": {first_context}, "second_context_bt": {second_context} }}
    """
    print(row.name, pun_word, first_meaning, second_meaning)
    try:
      response = get_response(prompt, model)
    except ValueError as e:
      print(f'Error: {e}')
      response = '{ "pun_word_bt": "ERROR", "first_meaning_bt": [], "second_meaning_bt": [], "first_context_bt": [], "second_context_bt": [] }'
      pass
    return response

  chunk_size = 100
  chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
  end = end if end > 0 else len(chunks)
  for i in range(start, end):
    if model == 'google':
      if translate_flag:
        chunks[i][['pun_word_fr', 'first_meaning_fr', 'second_meaning_fr', 'first_context_fr', 'second_context_fr']] = chunks[i].apply(google_translate, axis=1, args=('en', 'fr', '', '_fr'))
        save(chunks[i], f'{translate_dir}{model}/t/{i}.tsv')
      translate_df = load(f'{translate_dir}{model}/t/{i}.tsv')
      translate_df[['pun_word_bt', 'first_meaning_bt', 'second_meaning_bt', 'first_context_bt', 'second_context_bt']] = translate_df.apply(google_translate, axis=1, args=('fr', 'en', '_fr', '_bt'))
      save(translate_df, f'{translate_dir}{model}/{i}.tsv')
    else:
      if translate_flag:
        chunks[i][['pun_word_fr', 'first_meaning_fr', 'second_meaning_fr', 'first_context_fr', 'second_context_fr']] = chunks[i].apply(translate, axis=1)
        save(chunks[i], f'{translate_dir}{model}/t/{i}.tsv')
      translate_df = load(f'{translate_dir}{model}/t/{i}.tsv')
      translate_df[['pun_word_bt', 'first_meaning_bt', 'second_meaning_bt', 'first_context_bt', 'second_context_bt']] = translate_df.apply(back_translate, axis=1)
      save(translate_df, f'{translate_dir}{model}/{i}.tsv')


def google_translate(row, source_language_code, target_language_code, source_suffix, output_suffix):
  model = GoogleTranslateTransformer(project_id='gen-lang-client-0948849680')

  row_dict = row.to_dict()
  pun_word = row_dict[f'pun_word{source_suffix}']
  response = model.transform_documents(source_language_code=source_language_code, target_language_code=target_language_code, documents=[Document(page_content=pun_word)])
  response_pun_word = response[0].page_content

  first_meaning = ast.literal_eval(row_dict[f'first_meaning{source_suffix}'])
  response_first_meaning = []
  if len(first_meaning) > 0:
    documents = []
    for m in first_meaning:
      documents.append(Document(page_content=m))
    response = model.transform_documents(source_language_code=source_language_code, target_language_code=target_language_code, documents=documents)
    for r in response:
      response_first_meaning.append(r.page_content)

  second_meaning = ast.literal_eval(row_dict[f'second_meaning{source_suffix}'])
  response_second_meaning = []
  if len(second_meaning) > 0:
    documents = []
    for m in second_meaning:
      documents.append(Document(page_content=m))
    response = model.transform_documents(source_language_code=source_language_code, target_language_code=target_language_code, documents=documents)
    for r in response:
      response_second_meaning.append(r.page_content)

  first_context = ast.literal_eval(row_dict[f'first_context{source_suffix}'])
  response_first_context = []
  if len(first_context) > 0:
    documents = []
    for m in first_context:
      documents.append(Document(page_content=m))
    response = model.transform_documents(source_language_code=source_language_code, target_language_code=target_language_code, documents=documents)
    for r in response:
      response_first_context.append(r.page_content)

  second_context = ast.literal_eval(row_dict[f'second_context{source_suffix}'])
  response_second_context = []
  if len(second_context) > 0:
    documents = []
    for m in second_context:
      documents.append(Document(page_content=m))
    response = model.transform_documents(source_language_code=source_language_code, target_language_code=target_language_code, documents=documents)
    for r in response:
      response_second_context.append(r.page_content)

  response_json = {f"pun_word{output_suffix}": response_pun_word, f"first_meaning{output_suffix}": response_first_meaning, f"second_meaning{output_suffix}": response_second_meaning, f"first_context{output_suffix}": response_first_context, f"second_context{output_suffix}": response_second_context }
  print(row.name, pun_word, first_meaning, second_meaning)
  print(response_json)
  return pd.Series(response_json)


def get_cosine_similarity(df, model, start=0, end=-1):
  def apply(row, st_model):
    pun_word_embedding_en = st_model.encode([row['pun_word']], convert_to_tensor=True)
    first_meaning_embedding_en = torch.mean(
      st_model.encode(ast.literal_eval(row['first_meaning']), convert_to_tensor=True), dim=0, keepdim=True)
    second_meaning_embedding_en = torch.mean(
      st_model.encode(ast.literal_eval(row['second_meaning']), convert_to_tensor=True), dim=0, keepdim=True)

    pun_word_embedding_fr = st_model.encode([row['pun_word_fr']], convert_to_tensor=True)
    first_meaning_embedding_fr = torch.mean(
      st_model.encode(ast.literal_eval(row['first_meaning_fr']), convert_to_tensor=True), dim=0, keepdim=True)
    second_meaning_embedding_fr = torch.mean(
      st_model.encode(ast.literal_eval(row['second_meaning_fr']), convert_to_tensor=True), dim=0, keepdim=True)

    first_similarity_en = util.cos_sim(pun_word_embedding_en, first_meaning_embedding_en).item()
    second_similarity_en = util.cos_sim(pun_word_embedding_en, second_meaning_embedding_en).item()
    first_similarity_fr = util.cos_sim(pun_word_embedding_fr, first_meaning_embedding_fr).item()
    second_similarity_fr = util.cos_sim(pun_word_embedding_fr, second_meaning_embedding_fr).item()

    first_similarity_diff = first_similarity_en - first_similarity_fr
    second_similarity_diff = second_similarity_en - second_similarity_fr

    print(row.name, row['pun_word'], row['pun_word_fr'], row['pun_type'])
    print('first en', first_similarity_en, 'fr', first_similarity_fr, 'diff', first_similarity_diff)
    print('second en', second_similarity_en, 'fr', second_similarity_fr, 'diff', second_similarity_diff)

    result = {'first_similarity_en': first_similarity_en, 'second_similarity_en': second_similarity_en,
              'first_similarity_fr': first_similarity_fr, 'second_similarity_fr': second_similarity_fr,
              'first_similarity_diff': first_similarity_diff, 'second_similarity_diff': second_similarity_diff}
    return pd.Series(result)

  st_model = get_model(model)
  chunk_size = 10
  chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
  if end == -1:
    end = len(chunks)
  for i in range(start, end):
    current_df = chunks[i]
    current_df[['first_similarity_en', 'second_similarity_en', 'first_similarity_fr', 'second_similarity_fr',
               'first_similarity_diff', 'second_similarity_diff']] = current_df.apply(apply, axis=1, args=(st_model,))
    save(current_df, f'{similarity_dir}{model}/{i}.tsv')


def check_french_homonyms(df, model, start=0, end=-1):
  def apply(row):
    pun_word_fr = row['pun_word_fr']
    first_meaning_fr = row['first_meaning_fr']
    second_meaning_fr = row['second_meaning_fr']

    schema = '{ "is_homonym": 1 or 0, "first_meaning_overlap": 1 or 0, "second_meaning_overlap": 1 or 0 }'

    prompt = f"""
      Question 1: Is the French word "{pun_word_fr}" a homonym? If yes, output 1, else output 0.
      Question 2: Does the semantic range of the word "{pun_word_fr}" overlap with the semantic range of the words in this list: {first_meaning_fr}? If yes, output 1, else output 0.
      Question 3: Does the semantic range of the word "{pun_word_fr}" overlap with the semantic range of the words in this list: {second_meaning_fr}? If yes, output 1, else output 0.

      Return the output of the questions as a properly formatted json using this schema: {schema}
    """

    print(row.name, row['pun_word_fr'],  row['first_meaning_fr'], row['second_meaning_fr'])
    try:
      response = get_response(prompt, model)
    except ValueError as e:
      print(f'Error: {e}')
      response = '{ "is_homonym": -1, "first_meaning_overlap": -1, "second_meaning_overlap": -1 }'
      pass
    return response

  chunk_size = 10
  chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
  if end == -1:
    end = len(chunks)
  for i in range(start, end):
    chunks[i][['is_homonym', 'first_meaning_overlap', 'second_meaning_overlap']] = chunks[i].apply(apply, axis=1)
    save(chunks[i], f'{homonym_dir}{model}/{i}.tsv')


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
  translate_flag = False if len(sys.argv) > 5 else True

  if task == 'identify':
    df = load(cleaned_en_path)
    identify_pun_meanings(df, model, start, end)

  if task == 'translate':
    df = load_all(f'{identify_dir}gemini_pro/')
    save(df, f'{identify_dir}gemini_pro.tsv')
    translate_pun_meanings(df, model, start, end, translate_flag)

  if task == 'similarity':
    df = load_all(f'{translate_dir}o4/t/')
    save(df, f'{translate_dir}o4.tsv')
    get_cosine_similarity(df, model, start, end)

  if task == 'homonym':
    df = load_all(f'{similarity_dir}bilingual/')
    save(df, f'{similarity_dir}bilingual.tsv')
    check_french_homonyms(df, model, start, end)

  # if task == 'translate':
  #   df = load(f'{translate_dir}{model}/t/{start}.tsv')
  #   translate_pun_meanings(df, model, start, end)


    # for i in range(len(df)):
    #   json_str = df[['pun_word', 'first_meaning', 'second_meaning', 'first_context', 'second_context']].iloc[i]
    #   print(json_str)

  # find_synonyms()
  # translate_pun_meanings()
  # find_phonetically_similar_matches()
  # generate_french_puns()


