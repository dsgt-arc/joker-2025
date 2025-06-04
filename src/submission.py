import csv
import json
from io import StringIO

def tsv_to_json(tsv_content):
  f = StringIO(tsv_content)
  reader = csv.DictReader(f, delimiter='\t')

  output_list = []

  for row in reader:
    json_object = {
      "run_id": "dsgt_task_2_simple_mistral_medium",
      "manual": 0,
      "id_en": row.get('id_en'),
      "en": row.get('text_clean'),
      "fr": row.get('generated_pun')
    }
    print(json_object)
    output_list.append(json_object)

  return json.dumps(output_list, indent=4)

with open('../data/processed/regenerate/mistral.tsv', 'r', newline='', encoding='utf-8') as file:
    tsv_file_content = file.read()
json_output_from_file = tsv_to_json(tsv_file_content)
with open('../data/processed/submission/baseline/mistral.json', 'w', encoding='utf-8') as outfile:
    outfile.write(json_output_from_file)
