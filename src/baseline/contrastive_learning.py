from langchain_openai import ChatOpenAI
from data import load, load_all, save
from config import contrastive_dir, contrastive_path, openai_key, cleaned_fr_path
import pandas as pd


def create_non_puns(df):
  llm = ChatOpenAI(model='gpt-4o', api_key=openai_key)

  def apply_create_non_puns(row):
    pun = row['text_clean']
    is_pun = 1

    for i in range(10):
      if is_pun == 1:
        prompt = f"Input text: {pun}\n Return an output text that is the same length and similar in content. The output text must not contain any pun or homonym and must not be funny. Only return the output text."
        non_pun = llm.invoke(prompt).content

        prompt = f"Input text: {non_pun}\n If the input text contains a pun or homonym return 1, else return 0. Only return a single number (1 or 0)."
        is_pun = int(llm.invoke(prompt).content)
        pun = non_pun
        print(row.name, is_pun, non_pun)

    if is_pun == 1:
      prompt = f"Input: {non_pun}\n Translate into English. Output only the English translation."
      back_translation = llm.invoke(prompt).content
      print('------------- IS PUN -------------', back_translation)

    back_translation = ''
    if row.name % 10 == 0:
      prompt = f"Input: {non_pun}\n Translate into English. Output only the English translation."
      back_translation = llm.invoke(prompt).content
      print('------------ TRANSLATE -----------', back_translation)

    return pd.Series({'row': row.name, 'is_pun': is_pun, 'non_pun': non_pun, 'back_translation': back_translation})

  chunk_size = 100
  start = 14
  chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
  for i in range(start, len(chunks)):
    chunks[i][['row', 'is_pun', 'non_pun', 'back_translation']] = chunks[i].apply(apply_create_non_puns, axis=1)
    save(chunks[i], f'{contrastive_dir}{i}.csv')


def combine_files():
  return load_all(contrastive_dir)


def format_dataset(df):
  puns = df[['id_en', 'row', 'text_clean']]
  puns = puns.assign(target=1)
  # puns = puns.assign(is_pun=1)
  non_puns = df[['id_en', 'row']]
  non_puns['text_clean'] = df['non_pun']
  non_puns = non_puns.assign(target=0)
  # non_puns['is_pun'] = df['is_pun']
  concatenated = pd.concat([puns, non_puns])
  sorted = concatenated.sort_values(by=['row', 'target'])
  return sorted


def indentify_puns(df):
  llm = ChatOpenAI(model='gpt-4o', api_key=openai_key)

  def identify(row):
    input_fr = row['text_fr']
    prompt = f"Input: {input_fr}\n If the input contains a pun return 1, else return 0. Only return a single number (1 or 0) and explain your reasoning in 1 sentence."
    response = llm.invoke(prompt)
    print(row.name, response.content)
    return pd.Series({'pun': response.content})

  df[['pun']] = df.apply(identify, axis=1)
  return df


def identify_is_pun_true(df):
  return df[(df['target'] == 0) & (df['is_pun'] == 1)]


def get_average_lengths(df):
  df['avg_length'] = df['text_clean'].str.len()
  return df.groupby('target')['avg_length'].mean()


if __name__ == "__main__":
  # text_fr_df = load(cleaned_fr_path) #.head(5)
  # create_non_puns(text_fr_df)
  contrastive_df = combine_files()
  contrastive_df = format_dataset(contrastive_df)
  save(contrastive_df, contrastive_path)
  print(contrastive_df)
  print(get_average_lengths(contrastive_df))
  # print('count', identify_is_pun_true(contrastive_df))
