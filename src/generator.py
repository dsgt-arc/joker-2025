from sentence_transformers import SentenceTransformer, util
import torch
from utils import get_response
from config import generate_dir, translate_dir
from data import load, load_all, save


def generate_french_puns(df, model, start=0, end=-1):
    def apply(row):
        text_clean = row['text_clean']
        schema = '{ "generated_pun": "Generated French pun sentence" }'
        prompt = f"""
      English pun sentence: {text_clean}
      
      Pun word: {pun_word}
      First meaning of the pun_word: {first_meaning}
      Second meaning of the pun word: {second_meaning}
      Context words that support the first meaning: {first_context}
      Context words that support the second meaning: {second_meaning}
      
      Pun word translated into French: {pun_word_fr}
      First meaning translated into French: {first_meaning_fr}
      Second meaning translated into French: {second_meaning_fr}
      First context words translated into French: {first_context_fr}
      Second context words translated into French: {second_context_fr}

      Using the above information, generate a French pun sentence similar to the English pun sentence. Do not translate the English pun sentence literally. Be sure to find and use a French homonym that a native French speaker would find funny.
      
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

    chunk_size = 10
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    if end == -1:
        end = len(chunks)
    for i in range(start, end):
        chunks[i][['generated_pun']] = chunks[i].apply(apply, axis=1)
        save(chunks[i], f'{generate_dir}{model}/{i}.tsv')


if __name__ == "__main__":
  task = sys.argv[1]
  model = sys.argv[2]
  start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
  end = int(sys.argv[4]) if len(sys.argv) > 4 else -1
  translate_flag = False if len(sys.argv) > 5 else True

  if task == 'generate':
    df = load_all(f'{translate_dir}o4/t/')
    save(df, f'{translate_dir}o4.tsv')
    generate_french_puns(df, model, start, end)
