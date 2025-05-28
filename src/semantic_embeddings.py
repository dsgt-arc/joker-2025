import semantic_embeddings.util

if __name__ == "__main__":
  fasttext.util.download_model('en', if_exists='ignore')  # English
  fasttext.util.download_model('fr', if_exists='ignore')  # French
  ft_en = fasttext.load_model('cc.en.300.bin')
  ft_fr = fasttext.load_model('cc.fr.300.bin')