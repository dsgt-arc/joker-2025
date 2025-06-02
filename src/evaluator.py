import ast
import sys

import numpy as np
import pandas as pd
from config import identify_dir, translate_dir
from data import combine_en, load, load_all, save
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils import get_response

from sentence_transformers import SentenceTransformer, util
import torch


def evaluate_pun_location(df):
  y_true = df['manual_location'].str.lower()
  y_pred = df['pun_word'].str.lower()

  accuracy = accuracy_score(y_true, y_pred)
  precision = precision_score(y_true, y_pred, average='weighted', zero_division=np.nan)
  recall = recall_score(y_true, y_pred, average='weighted', zero_division=np.nan)
  f1 = f1_score(y_true, y_pred, average='weighted', zero_division=np.nan)
  print('pun_location')
  print('accuracy:', accuracy)
  print('precision:', precision)
  print('recall:', recall)
  print('f1-score:', f1, '\n')


def evaluate_pun_type(df):
  y_true = df['manual_type'].str.lower()
  y_pred = df['pun_type'].str.lower()

  accuracy = accuracy_score(y_true, y_pred)
  precision = precision_score(y_true, y_pred, average='weighted', zero_division=np.nan)
  recall = recall_score(y_true, y_pred, average='weighted', zero_division=np.nan)
  f1 = f1_score(y_true, y_pred, average='weighted', zero_division=np.nan)
  print('pun_type')
  print('accuracy:', accuracy)
  print('precision:', precision)
  print('recall:', recall)
  print('f1-score:', f1, '\n')


def evaluate_alternative_words(df, prompt_llm):
  def apply(row):
    manual_alternative = row['manual_alternative'].lower()
    generated_alternative = row['pun_alternative'].lower()

    print(row.name, manual_alternative, generated_alternative)
    if manual_alternative == generated_alternative:
      print('{"bool": 1}')
      return pd.Series({"bool": 1})

    schema = '{ "bool": 0 or 1 }'
    prompt = f"""
      Does the semantic range of "{generated_alternative}" overlap with the semantic range of "{manual_alternative}"? If yes return 1, else return 0.
      Return the output as a json using this schema: {schema}
    """
    return get_response(prompt, 'gpt-4o')

  if prompt_llm:
    df['evaluated_alternative'] = df.apply(apply, axis=1)
    save(df, identification_gpt_4o_path)

  loaded_df = load(identification_gpt_4o_path)
  y_true = loaded_df['manual_alternative'].str.lower()
  y_pred = loaded_df.apply(lambda row: row['manual_alternative'].lower() if row['evaluated_alternative'] else 'false', axis=1)

  accuracy = accuracy_score(y_true, y_pred)
  precision = precision_score(y_true, y_pred, average='weighted', zero_division=np.nan)
  recall = recall_score(y_true, y_pred, average='weighted', zero_division=np.nan)
  f1 = f1_score(y_true, y_pred, average='weighted', zero_division=np.nan)
  print('pun_alternative')
  print('accuracy:', accuracy)
  print('precision:', precision)
  print('recall:', recall)
  print('f1-score:', f1, '\n')


def evaluate_translations(df):
  model_name = 'all-MiniLM-L6-v2'
  model = SentenceTransformer(model_name)
  combined = []
  total_problems = []
  def apply(row):
    source = [row['pun_word']] + row['first_meaning'] + row['second_meaning']
    back_translated = [row['pun_word_bt']] + row['first_meaning_bt'] + row['second_meaning_bt']

    problems = 0
    source_embeddings = model.encode(source, convert_to_tensor=True)
    back_translated_embeddings = model.encode(back_translated, convert_to_tensor=True)
    similarities = []
    for i in range(len(source_embeddings)):
      if i < len(source) and i < len(back_translated) and source[i] == back_translated[i]:
        similarities.append(1)
      elif i < len(source_embeddings) and i < len(back_translated_embeddings):
        similarities.append(util.cos_sim(source_embeddings[i], back_translated_embeddings[i]).item())
      else:
        problems += 1
    similarity = sum(similarities) / len(similarities)

    # print(row['pun_word'], row['pun_word_bt'], similarity)
    combined.append(similarity)
    total_problems.append(problems)


  df['first_meaning'] = df['first_meaning'].apply(ast.literal_eval)
  df['second_meaning'] = df['second_meaning'].apply(ast.literal_eval)
  df['first_meaning_bt'] = df['first_meaning_bt'].apply(ast.literal_eval)
  df['second_meaning_bt'] = df['second_meaning_bt'].apply(ast.literal_eval)
  df[['pun_word', 'first_meaning', 'second_meaning', 'pun_word_bt', 'first_meaning_bt', 'second_meaning_bt']].apply(apply, axis=1)
  print('mean cosine similarity', np.mean(combined))
  print('variance', np.var(combined))
  print('top quartile', len([x for x in combined if x > 0.75]), len([x for x in combined if x > 0.75]) / len(df))
  print('bottom quartile', len([x for x in combined if x < 0.25]), len([x for x in combined if x < 0.25]) / len(df))
  print('problems', sum(total_problems), sum(total_problems) / len(df))


if __name__ == "__main__":
  task = sys.argv[1]
  model = sys.argv[2]

  if task == 'identify':
    df = load_all(f'{identify_dir}{model}/')
    save(df, f'{identify_dir}{model}.tsv')
    df = load(f'{identify_dir}{model}.tsv')
    df = df[df['manual_location'].str.len() > 0]
    print('row count', len(df))
    evaluate_pun_location(df)
    evaluate_pun_type(df)
    # evaluate_alternative_words(df, prompt_llm=True
    
  if task == 'translate':
    df = load_all(f'{translate_dir}{model}/')
    save(df, f'{translate_dir}{model}.tsv')
    df = load(f'{translate_dir}{model}.tsv')
    df = df[df['pun_word_bt'].str.len() > 0]
    print('row count', len(df))
    evaluate_translations(df)
