import numpy as np
from config import identification_gpt_4o_path
from data import load, save
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def evaluate_pun_location(df):
  y_true = df['manual_location'].str.lower()
  y_pred = df['pun_location'].str.lower()

  accuracy = accuracy_score(y_true, y_pred)
  precision = precision_score(y_true, y_pred, average='weighted', zero_division=np.nan)
  recall = recall_score(y_true, y_pred, average='weighted', zero_division=np.nan)
  f1 = f1_score(y_true, y_pred, average='weighted', zero_division=np.nan)
  print('\npun_location')
  print('accuracy:', accuracy)
  print('precision:', precision)
  print('recall:', recall)
  print('f1-score:', f1)


def evaluate_pun_type(df):
  y_true = df['manual_type'].str.lower()
  y_pred = df['pun_type'].str.lower()

  accuracy = accuracy_score(y_true, y_pred)
  precision = precision_score(y_true, y_pred, average='weighted', zero_division=np.nan)
  recall = recall_score(y_true, y_pred, average='weighted', zero_division=np.nan)
  f1 = f1_score(y_true, y_pred, average='weighted', zero_division=np.nan)
  print('\npun_type')
  print('accuracy:', accuracy)
  print('precision:', precision)
  print('recall:', recall)
  print('f1-score:', f1)


def evaluate_alternative_words(df):
  return True


if __name__ == "__main__":
  df = load(identification_gpt_4o_path)
  print('model: gpt-4o')
  evaluate_pun_location(df)
  evaluate_pun_type(df)
  evaluate_alternative_words(df)
