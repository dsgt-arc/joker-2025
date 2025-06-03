from sentence_transformers import SentenceTransformer, util
import torch
import sys
from utils import get_response
from config import contrastive_dir, generate_dir, translate_dir
from data import load, load_all, save


def generate_french_puns(df, model, start=0, end=-1):
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

  if task == 'contrastive':
    df = load_all(f'{contrastive_dir}baseline/gemini/{model}/')
    save(df, f'{contrastive_dir}baseline/gemini/{model}.tsv')
    regenerate_failed_puns(df, model, start, end)
