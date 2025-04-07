import glob
import os
import pandas as pd
import re
from config import cleaned_en_path, cleaned_fr_path, combined_fr_path, location_fr_input_path, location_fr_qrels_path, translation_en_path, translation_fr_path


def load(path):
  ext = os.path.splitext(path)[1]
  if ext == '.json':
    data = pd.read_json(path)
  elif ext == '.csv':
    data = pd.read_csv(path)
  else:
    with open(path, 'r') as file:
      data = file.read()

  print(f'Loaded {path}')
  if type(data) is pd.DataFrame:
    print(f'Row count:', len(data))
  return data.fillna('')


def load_all(path):
  files = glob.glob(os.path.join(path, "*.csv"))
  print('##########', files)
  df_list = []

  for file in files:
    df = pd.read_csv(file)
    df_list.append(df)

  combined_df = pd.concat(df_list, ignore_index=True)
  print(f'Loaded all csvs in {path}')
  print(f'Row count:', len(combined_df))
  return combined_df.fillna('')


def save(data, path):
  dir = os.path.dirname(path)
  if not os.path.exists(dir):
    os.makedirs(dir)

  ext = os.path.splitext(path)[1]
  if ext == '.csv':
    data.to_csv(path, index=False)
  else:
    with open(path, 'w') as file:
      data.write(file)

  print(f'Saved {path}')


def clean(text):
  text = text.str.replace("''", '"').str.replace(' - ', '-').str.replace('",', ',"')
  text = text.str.replace('3. ', '3.').str.replace(".'.", ".'").str.replace(': )', '.')

  def lower_old_puns(sentence):
    words = sentence.split()
    if words and words[0] == "OLD":
      all_caps_sequence = True
      for i in range(1, len(words)):
        word = words[i]
        if len(word) <= 2:
          continue
        if not word.isupper():
          all_caps_sequence = False
        if word.isupper() and all_caps_sequence:
          words[i] = word.lower()
      words[0] = words[0].capitalize()
    return ' '.join(words)
  text = text.apply(lower_old_puns)

  tom_dict = {'said Tom': 'Tom said', 'spouted Tom': 'Tom spouted', 'asked Tom': 'Tom asked',
              'yelled Tom': 'Tom yelled', 'reported Tom': 'Tom reported', "cried Tom's band": "Tom's band cried"}
  text = text.replace(tom_dict, regex=True)

  punctuation = [',', '!', '?']
  for p in punctuation:
    text = text.apply(lambda x: f'"{x}'.replace(f'{p} Tom', f'{p}" Tom') if f'{p} Tom' in x else x)

  tom_dict = {"Tom's": 'his', 'Tom': 'he'}
  text = text.replace(tom_dict, regex=True)

  def replace_and_capitalize(sentence):
    return re.sub(r' -- (\w)', lambda match: '. ' + match.group(1).upper(), sentence)
  text = text.apply(replace_and_capitalize)

  return text


def remove_quotes(text):
  print('text', text)
  print('clean', text.replace('"', ''))
  return text.replace('"', '')


def combine(translation, location_input, location_qrels):
  combined_df = pd.merge(translation, location_input, left_on='text_fr', right_on='text', how='left')
  combined_df = pd.merge(combined_df, location_qrels, left_on='id', right_on='id', how='left')
  combined_df = combined_df[['id_en', 'text_fr', 'id', 'location']]
  return combined_df.groupby('id_en').agg(lambda x: list(x.dropna())).reset_index()


def clean_en():
  translation_en_df = load(translation_en_path)
  translation_en_df['text_clean'] = clean(translation_en_df['text_en'])
  save(translation_en_df, cleaned_en_path)


def clean_fr():
  translation_fr_df = load(translation_fr_path)
  translation_fr_df['text_clean'] = translation_fr_df['text_fr'].apply(remove_quotes)
  save(translation_fr_df, cleaned_fr_path)


def combine_fr():
  translation_fr_df = load(translation_fr_path)
  location_fr_input_df = load(location_fr_input_path)
  location_fr_qrels_df = load(location_fr_qrels_path)

  translation_fr_df = combine(translation_fr_df, location_fr_input_df, location_fr_qrels_df)
  save(translation_fr_df, combined_fr_path)


if __name__ == "__main__":
  clean_en()
  clean_fr()
  combine_fr()
