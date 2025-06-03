import configparser
import os

config = configparser.ConfigParser()
config.read('../config.ini')

openai_key = os.environ.get("OPENAI_API_KEY")
gemini_key = os.environ.get("GEMINI_API_KEY")
anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
mistral_key = os.environ.get("MISTRAL_API_KEY")
openrouter_key = os.environ.get("OPENROUTER_API_KEY")

claude = config['model']['claude']
deepseek = config['model']['deepseek']
gemini = config['model']['gemini']
gemini_pro = config['model']['gemini_pro']
gpt = config['model']['gpt']
mistral = config['model']['mistral']
o3 = config['model']['o3']
o4 = config['model']['o4']
google = config['model']['google']
camembert = config['model']['camembert']
bilingual = config['model']['bilingual']

translation_path = config['path']['translation']
translation_en_path = config['path']['translation_en']
translation_fr_path = config['path']['translation_fr']

location_en_input_path = config['path']['location_en_input']
location_fr_input_path = config['path']['location_fr_input']
location_en_qrels_path = config['path']['location_en_qrels']
location_fr_qrels_path = config['path']['location_fr_qrels']
location_manual_path = config['path']['location_manual']

cleaned_en_path = config['path']['cleaned_en']
cleaned_fr_path = config['path']['cleaned_fr']
combined_en_path = config['path']['combined_en']
combined_fr_path = config['path']['combined_fr']

fasttext_en_path = config['path']['fasttext_en']
fasttext_fr_path = config['path']['fasttext_fr']
fasttext_align_en_path = config['path']['fasttext_align_en']
fasttext_align_fr_path = config['path']['fasttext_align_fr']

contrastive_path = config['path']['contrastive']

identify_dir = config['dir']['identify']
translate_dir = config['dir']['translate']
similarity_dir = config['dir']['similarity']
homonym_dir = config['dir']['homonym']
generate_dir = config['dir']['generate']
contrastive_baseline_dir = config['dir']['contrastive_baseline']
contrastive_dir = config['dir']['contrastive']


