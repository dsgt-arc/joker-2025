from sentence_transformers import SentenceTransformer, util
import torch
import sys
from utils import get_model, get_response
from config import contrastive_dir, generate_dir, regenerate_dir, translate_dir
from data import load, load_all, save
import pandas as pd


def generate(df, model_str, start=0, end=-1):
    def apply(row):
        text_clean = row['text_clean']
        schema = '{ "generated_pun": "Generated French pun sentence" }'
        prompt = f"""
        Here is an English pun: {text_clean}
        
        Generate a similar French pun. Use a homonym where its first meaning is related to the broader context and its second meaning is part of an idiomatic phrase. Both meanings should be obvious and funny to a native French speaker.

        Return the output as a properly formatted json using this schema: {schema}
        """

        print(row.name, text_clean)
        try:
            response = get_response(prompt, model)
        except ValueError as e:
            print(f'Error: {e}')
            response = '{ "generated_pun": "ERROR" }'
            pass
        return response

    model = get_model(model_str)
    chunk_size = 10
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    if end == -1:
        end = len(chunks)
    for i in range(start, end):
        chunks[i][['generated_pun']] = chunks[i].apply(apply, axis=1)
        save(chunks[i], f'{generate_dir}{model_str}/{i}.tsv')


def regenerate(df, contrastive_df, model_str, start=0, end=-1):
  def create_context_string(row):
    text = row['text_clean']
    target = row['target']
    prefix = 'Contains a pun: ' if target == 1 else 'Does not contain a pun: '
    return prefix + text

  pun_df = contrastive_df[contrastive_df['target'] == 1].sample(n=25)
  non_pun_df = contrastive_df[contrastive_df['target'] == 0].sample(n=25)
  context_df = pd.concat([pun_df, non_pun_df], axis=0)
  context_df['string'] = context_df.apply(create_context_string, axis=1)
  context = '\n'.join(context_df['string'].tolist())
  print(context)
  model = get_model(model_str)
  eval_model = get_model('gemini')

  def apply(row):
    def evaluate(generated_pun):
      schema = '{ "is_pun": 0 or 1 }'
      prompt = f"""
        {context}
        Input: {generated_pun}
        If the input contains a pun return 1, else return 0, in a properly formatted json using this schema: {schema}
      """
      print(prompt)
      try:
        print(row.name, generated_pun)
        is_pun = get_response(prompt, eval_model)['is_pun']
        if is_pun == 1:
          return True
      except ValueError as e:
        print(f'Error: {e}')
        return False
      return False

    if row['is_pun'] == 0:
      text_clean = row['text_clean']
      schema = '{ "generated_pun": "Generated French pun sentence" }'
      prompt = f"""
          Here is an English pun: {text_clean}
  
          Generate a similar French pun. Use a homonym where its first meaning is related to the broader context and its second meaning is part of an idiomatic phrase. Both meanings should be obvious and funny to a native French speaker.
  
          Return the output as a properly formatted json using this schema: {schema}
          """

      for i in range(10):
        try:
          print(row.name, i, text_clean, row['is_pun'])
          generated_pun = get_response(prompt, model)['generated_pun']
          if evaluate(generated_pun):
            return pd.Series({"generated_pun": generated_pun, "is_pun": "1"})
        except ValueError as e:
          print(f'Error: {e}')
          pass

    return pd.Series({"generated_pun": row['generated_pun'], "is_pun": "0"})

  chunk_size = 10
  chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
  if end == -1:
    end = len(chunks)
  for i in range(start, end):
    chunks[i][['generated_pun', 'is_pun']] = chunks[i].apply(apply, axis=1)
    save(chunks[i], f'{regenerate_dir}{model_str}/{i}.tsv')


if __name__ == "__main__":
  task = sys.argv[1]
  model = sys.argv[2]
  start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
  end = int(sys.argv[4]) if len(sys.argv) > 4 else -1
  translate_flag = False if len(sys.argv) > 5 else True

  if task == 'generate':
    df = load_all(f'{translate_dir}o4/t/')
    save(df, f'{translate_dir}o4.tsv')
    generate(df, model, start, end)

  if task == 'regenerate':
    contrastive_df = load(f'{contrastive_dir}dataset.csv')
    print('contrastive count', len(contrastive_df))

    # df = load_all(f'{contrastive_dir}baseline/gemini/{model}/')
    df = load(f'{contrastive_dir}baseline/gemini/{model}.tsv')
    df = df[df['is_pun'] == 0]
    regenerate(df, contrastive_df, model, start, end)


  def generate_cot(df, model_str, start=0, end=-1):
    def apply(row):
      text_clean = row['text_clean']
      schema = '{ "generated_pun": "Generated French pun sentence" }'
      prompt = f"""
        Is the word "{row['pun_word_fr']}" a homonym?
        
      
      
        Here is an English pun: {text_clean}
        
        French pun word with two meanings: {row['pun_word_fr']}
        French first meaning of the pun word: {row['first_meaning']}
        French second meaning of the pun word: {row['second_meaning']}
        
        Generate a similar French pun.
        
        Pun word with two meanings: {row['pun_word']}
        First meaning of the pun word: {row['first_meaning_fr']}
        Second meaning of the pun word: {row['second_meaning_fr']}
        
        

        Generate a similar French pun. Use a homonym where its first meaning is related to the broader context and its second meaning is part of an idiomatic phrase. Both meanings should be obvious and funny to a native French speaker.

        Return the output as a properly formatted json using this schema: {schema}
        """

      print(row.name, text_clean)
      try:
        response = get_response(prompt, model)
      except ValueError as e:
        print(f'Error: {e}')
        response = '{ "generated_pun": "ERROR" }'
        pass
      return response

    model = get_model(model_str)
    chunk_size = 10
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    if end == -1:
      end = len(chunks)
    for i in range(start, end):
      chunks[i][['generated_pun']] = chunks[i].apply(apply, axis=1)
      save(chunks[i], f'{generate_dir}{model_str}/{i}.tsv')
