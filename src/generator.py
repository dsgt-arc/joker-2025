from data import load, save
from config import openai_key, combined_en_path, identification_gpt_4o_path
from utils import get_response
from embeddings import read_faiss_index, retrieve_similar_words, load_embedding_matrix

def identify_pun_meanings(df, model):
  def apply(row):
    text_clean = row['text_clean']
    schema = '{ "pun_word": "pun_word", "pun_type": "pun_type", "alternative_word": "alternative_word", "idiomatic_phrase": "idiomatic_phrase" }'
    prompt = f"""
      Text: {text_clean}
    
      Step 1: Identify the pun word in this text. Output one word.
      Step 2: Is this is a homographic or homophonic pun. Output either the word "homophonic" or the word "homographic".
      Step 3: If the pun is homographic, identify the alternative meaning of the pun word that makes the text funny. Or, if the pun is homophonic, identify the alternative word that the pun word alludes to. Output one word.
      Step 4: Pun words often occur within idiomatic phrases that support the alternative meaning. Identify the idiomatic phrase that makes the pun funny. Output a short phrase.
      
      Return the output of the steps as a json using this schema: {schema}
    """
    print(row.name, text_clean)
    return get_response(prompt, model)

  df[['generated_location', 'generated_type', 'generated_alternative', 'phrase']] = df.apply(apply, axis=1)
  return df


def find_synonyms(df, model):
  # TODO: get lists of synonyms for each of the two meanings of the pun word
  return True


def translate_pun_meanings(df, model):
  # TODO: translate all of the synonyms, not just the two meanings
  def apply(row):
    pun_word = row['pun_word_en']
    alternative_word = row['alternative_word_en']
    idiomatic_phrase = row['phrase_en']
    schema = '{"pun_word": "step_1_word", "alternative_word": "step_2_word", "idiomatic_phrase": "step_3_phrase"}'

    prompt = f"""
      Step 1: Translate the word "{pun_word}" into French. Output one word.
      Step 2: Translate the word "{alternative_word}" into French. Output one word.
      Step 3: Translate this idiomatic phrase into French: {idiomatic_phrase}. Do not translate literally. Output a short French idiomatic phrase.
      
      Return the output of the steps as a json using this schema: {schema}
    """
    print(row.name, pun_word, alternative_word, idiomatic_phrase)
    return get_response(prompt, model)

  df[['pun_word_fr', 'alternative_word_fr', 'phrase_fr']] = df.apply(apply, axis=1)
  return df


def find_phonetically_similar_matches(df):
  # TODO: Use phonetic embeddings to find phonetically similar matches
    index = read_faiss_index()
    embedding_matrix = load_embedding_matrix()

    def apply(row):
      alternative_word_fr = row['alternative_word_fr']
      return retrieve_similar_words(index, embedding_matrix, query_word=alternative_word_fr, top_k=5)
    
    df['similar_words'] = df.apply(apply, axis=1)
    return df

def generate_french_puns(df):
  # TODO: generate French puns using the information generated in previous steps
  return True
  # def prompt_llm(row):
  #   text_clean = row['text_clean']
  #   pun_type = row['pun_type']
  #   pun_word_fr = row['pun_word_fr']
  #   alternative_meaning_fr = row['alternative_meaning_fr']
  #
  #   schema = '{"is_homonym": 1 or 0, "meanings_overlap": 1 or 0}'
  #
  #   prompt = f"""
  #     Question 1: Is the French word "{pun_word_fr}" a homonym? If yes, output 1, else output 0.
  #     Question 2: Does the semantic range of the word "{pun_word_fr}" overlap with the semantic range of the word "{alternative_meaning_fr}"? If yes, output 1, else output 0.
  #
  #     Return the output of the steps as a json in this format (Do not include any additional text): {schema}
  #   """
  #   response = llm.invoke(prompt).content
  #   print(row.name, response[response.find('{'):response.rfind('}') + 1].replace('\n', ''))
  #   response_json = json.loads(response[response.find('{'):response.rfind('}') + 1])
  #
  #   if response_json['is_homonym'] == 1 and response_json['meanings_overlap'] == 1:
  #
  #
  #   is_pun = int(llm.invoke(prompt).content)
  #   pun = non_pun
  #   print(row.name, is_pun, non_pun)
  #
  #
  #
  #   schema = '{ "generated_french_pun": "generated_french_pun" }'
  #   prompt = f"""
  #     Step 1: Identify the pun word in this text. Output one word.
  #     Step 2: Is this is a homographic or homophonic pun. Output either the word "homophonic" or the word "homographic".
  #     Step 3: If the pun is homographic, identify the alternative meaning of the pun word that makes the text funny. Output one word.
  #     Step 4: If the pun is homophonic, identify the alternative word that the pun word alludes to. Output one word.
  #     Step 5: Pun words often occur within idiomatic phrases that support the alternative meaning. Identify the idiomatic phrase that makes the pun funny. Output a short phrase.
  #
  #     Return the output of the steps as a json in this format (Do not include any additional text): {schema}
  #   """
  #
  #   response = llm.invoke(prompt).content
  #   print(row.name, response[response.find('{'):response.rfind('}') + 1].replace('\n', ''))
  #   response_json = json.loads(response[response.find('{'):response.rfind('}') + 1])
  #   return pd.Series(response_json)
  #
  # df[['pun_word_en', 'pun_type', 'alternative_meaning_en', 'alternative_word_en', 'phrase_en']] = df.apply(prompt_llm,
  #                                                                                                          axis=1)
  # return df


if __name__ == "__main__":
  model = 'gpt-4o'

  df = load(combined_en_path)
  df = identify_pun_meanings(df, model)
  save(df, identification_gpt_4o_path)

  # find_synonyms()
  # translate_pun_meanings()
  # find_phonetically_similar_matches()
  # generate_french_puns()


