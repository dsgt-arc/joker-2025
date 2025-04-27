import configparser

config = configparser.ConfigParser()
config.read('../config.ini')

openai_key = config['api-key']['openai']
anthropic_key = config['api-key']['anthropic']
deepseek_key = config['api-key']['deepseek']

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

contrastive_path = config['path']['contrastive']
contrastive_dir = config['path']['contrastive_dir']

identification_gpt_4o_path = config['path']['identification_gpt_4o']
