import faiss
import pandas as pd
import numpy as np


def create_faiss_index(embedding_matrix_path="data/embedding_matrix.npy", save_path="data/faiss.index"):
  embedding_matrix = np.load(embedding_matrix_path)
  faiss.normalize_L2(embedding_matrix)
  index = faiss.IndexFlatIP(embedding_matrix.shape[1])
  index.add(embedding_matrix)
  faiss.write_index(index, save_path)
  return index, embedding_matrix


def read_faiss_index(index_path="data/faiss.index"):
  index = faiss.read_index(index_path)
  return index


def load_embedding_matrix(embedding_matrix_path="data/embedding_matrix.npy"):
  embedding_matrix = np.load(embedding_matrix_path)
  return embedding_matrix


def retrieve_similar_words(index, embedding_matrix, query_word, top_k=5,
                           converted_phrases_path="data/converted_phrases.csv"):
  df = pd.read_csv(converted_phrases_path)
  words = df["word"].tolist()

  word_to_idx = {word: i for i, word in enumerate(words)}

  if query_word not in word_to_idx:
    raise ValueError(f"'{query_word}' not found in vocabulary.")

  query_idx = word_to_idx[query_word]
  query_emb = embedding_matrix[query_idx].reshape(1, -1)
  faiss.normalize_L2(query_emb)

  similarities, indices = index.search(query_emb, top_k + 1)

  results = []
  for j, i in enumerate(indices[0]):
    if i != query_idx:
      results.append((words[i], similarities[0][j]))

  return results[:top_k]


if __name__ == "__main__":
  create_faiss_index()
