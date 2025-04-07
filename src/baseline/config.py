import configparser

config = configparser.ConfigParser()
path = '../config/config.ini'

def write_config():
  config['api-key'] = {
    'openai': '', # Add OpenAI Key
    'anthropic': '', # Add Anthropic Key
    'deepseek': '' # Add Deepseek Key
  }
  config['path'] = {
    'translation_en': '../data/2024/translation/EN-FR-train/joker_translation_EN-FR_train_input.json',
    'translation_fr': '../data/2024/translation/EN-FR-train/joker_translation_EN-FR_train_qrels.json',
    'location_en_input': '../data/2023/location/en/joker_loc_interpret_EN_train_input.json',
    'location_fr_input': '../data/2023/location/fr/joker_loc_interpret_FR_train_input.json',
    'location_en_qrels': '../data/2023/location/en/joker_loc_interpret_EN_train_qrels.json',
    'location_fr_qrels': '../data/2023/location/fr/joker_loc_interpret_FR_train_qrels.json',
    'cleaned_en': '../data/processed/cleaned_en.csv',
    'cleaned_fr': '../data/processed/cleaned_fr.csv',
    'combined_fr': '../data/processed/combined_fr.csv',
    'contrastive': '../data/processed/contrastive.csv',
    'contrastive_dir': '../data/processed/contrastive/',
  }

  with open(path, 'w') as file:
    config.write(file)

  print(f'config written to {path}')


def read_config():
  config.read(path)
  return config


if __name__ == "__main__":
  write_config()

config = read_config()

openai_key = config['api-key']['openai']
anthropic_key = config['api-key']['anthropic']
deepseek_key = config['api-key']['deepseek']

translation_en_path = config['path']['translation_en']
translation_fr_path = config['path']['translation_fr']
location_en_input_path = config['path']['location_en_input']
location_fr_input_path = config['path']['location_fr_input']
location_en_qrels_path = config['path']['location_en_qrels']
location_fr_qrels_path = config['path']['location_fr_qrels']
cleaned_en_path = config['path']['cleaned_en']
cleaned_fr_path = config['path']['cleaned_fr']
combined_fr_path = config['path']['combined_fr']
contrastive_path = config['path']['contrastive']
contrastive_dir = config['path']['contrastive_dir']