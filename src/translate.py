from sentence_transformers import SentenceTransformer, util
import torch
from config import camembert

model = SentenceTransformer(camembert)

def get_similarity(row):
  pun_word = "pied"

  fr = {"pun_word_fr": "pied", "first_meaning_fr": ["unité de longueur", "mesure linéaire"],
   "second_meaning_fr": ["appendice", "extrémité", "patte"], "first_context_fr": ["10", "serpent"],
   "second_context_fr": ["serpents", "n'ont pas"]}
  pun_word = row['pun_word_fr']
  first_meaning = row['first_meaning_fr']
  second_meaning = row['second_meaning_fr']
  pun_word_embedding = model.encode([pun_word], convert_to_tensor=True)
  first_meaning_embedding =  torch.mean(model.encode(first_meaning, convert_to_tensor=True), dim=0, keepdim=True)
  second_meaning_embedding = torch.mean(model.encode(second_meaning, convert_to_tensor=True), dim=0, keepdim=True)


  cosine_scores = util.cos_sim(embedding_en, embedding_fr)
  similarity = cosine_scores.item()

  print(f"English word: {words_en}")
  print(f"French word: {words_fr}")
  print(f"Semantic Similarity: {similarity:.4f}")

get_similarity('hello cat', 'bonjour chat')