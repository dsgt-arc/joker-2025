import numpy as np
import pandas as pd
from config import identification_gpt_4o_path
from data import load, save
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils import get_response


def evaluate_pun_location(df):
  y_true = df['manual_location'].str.lower()
  y_pred = df['generated_location'].str.lower()

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
  y_pred = df['generated_type'].str.lower()

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
    generated_alternative = row['generated_alternative'].lower()

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


if __name__ == "__main__":
  df = load(identification_gpt_4o_path)
  print('model: gpt-4o\n')
  evaluate_pun_location(df)
  evaluate_pun_type(df)
  evaluate_alternative_words(df, prompt_llm=True)
