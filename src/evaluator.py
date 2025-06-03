import ast
import sys

import numpy as np
import pandas as pd
from config import contrastive_dir, generate_dir, identify_dir, translate_dir
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


def evaluate_generations(df, context_df, eval_model, start=0, end=-1):
  def create_context_string(row):
    text = row['text_clean']
    target = row['target']
    prefix = 'Contains a pun: ' if target == 1 else 'Does not contain a pun: '
    return prefix + text

  pun_df = context_df[context_df['target'] == 1].sample(n=25)
  non_pun_df = context_df[context_df['target'] == 0].sample(n=25)
  context_df = pd.concat([pun_df, non_pun_df], axis=0)
  context_df['string'] = context_df.apply(create_context_string, axis=1)
  context = '\n'.join(context_df['string'].tolist())

  def apply(row):
    text = row['generated_pun']
    schema = '{ "is_pun": 0 or 1 }'
    prompt = f"""
      {context}
      Input: {text}
      If the input contains a pun return 1, else return 0, in a properly formatted json using this schema: {schema}
    """
    print(row.name, text)
    try:
      response = get_response(prompt, eval_model)
    except ValueError as e:
      print(f'Error: {e}')
      response = '{ "is_pun": "ERROR" }'
      pass
    return response

  chunk_size = 10
  chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
  if end == -1:
    end = len(chunks)
  for i in range(start, end):
    chunks[i][['is_pun']] = chunks[i].apply(apply, axis=1)
    save(chunks[i], f'{contrastive_dir}baseline/{eval_model}/{model}/{i}.tsv')


if __name__ == "__main__":
  task = sys.argv[1]
  model = sys.argv[2]
  eval_model = sys.argv[3] if len(sys.argv) > 3 else ''
  start = int(sys.argv[4]) if len(sys.argv) > 4 else 0
  end = int(sys.argv[5]) if len(sys.argv) > 5 else -1

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

  if task == 'generate':
    context_df = load(f'{contrastive_dir}dataset.csv')
    print('context count', len(context_df))

    df = load_all(f'{generate_dir}{model}/')
    save(df, f'{generate_dir}{model}.tsv')
    print('generate count', len(df))
    evaluate_generations(df, context_df, eval_model, start, end)

  if task == 'gen_count':
    df = load_all(f'{contrastive_dir}baseline/o4/{model}/')
    save(df, f'{contrastive_dir}baseline/o4/{model}.tsv')
    print('eval_model=o4 - row count', len(df))
    print(df['is_pun'].value_counts(normalize=True))

    df = load_all(f'{contrastive_dir}baseline/gemini/{model}/')
    save(df, f'{contrastive_dir}baseline/gemini/{model}.tsv')
    print('\neval_model=gemini - row count', len(df))
    print(df['is_pun'].value_counts(normalize=True))


