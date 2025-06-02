import pandas as pd

df_cleaned_en = pd.read_csv("data/processed/cleaned_en.tsv", sep="\t")
df_cleaned_fr = pd.read_csv("data/processed/cleaned_fr.tsv", sep="\t")
df_combined = pd.DataFrame(
    {
        "text_clean": df_cleaned_en["text_clean"],
        "initial_translation": df_cleaned_fr["text_clean"],
        "id_en": df_cleaned_en["id_en"],
    }
)

df_combined.to_csv("data/processed/english_french_cleaned.tsv", sep="\t", index=False)
