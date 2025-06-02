from data import load, save
from utils import get_response_not_json
from config import identification_gpt_4o_path, refinement_gpt_4o_path
import re


evaluators = {
    "equivalence": {
        "instructions": """You are fluent in both English and French, with a deep understanding of humor in both languages. Your task is to compare a translated pun to its
          original version and determine whether the meanings of the source are maintained in the translation, and if the translation is still humorous.

Rate the translation 0 - 2 using one of the following categories:
    2: Full Equivalence: Both the literal and contextual meanings of the pun remain the same in the translation. The humor, wordplay, and intended effect are fully preserved.
    1: Part Equivalence: The contextual meaning of the pun is similar in both languages, but the literal meaning of the target word differs. While the translation remains
        metaphorical, the wordplay may be altered.
    0: Non-Equivalence: The contextual meaning is somewhat preserved, but the translation is no longer metaphorical. The literal meaning of the target words differs 
        significantly, resulting in a loss of the original wordplay.

Evaluate the pun based on these criteria and provide a justification in english for your rating. Provide your answer in the following format:

Rating: <number from 0 - 2>
Justification: <very concise explanation of key issues>"""
    },
    "mistranslation": {
        "instructions": """You are fluent in both English and French, with a deep understanding of humor in both languages. Your task is to compare a translated pun to its 
        original version and assess whether the literal 
        meaning of the pun's wordplay is similar in both languages, but the contextual meaning or intended humor is lost or altered in translation.

Rate the translation 0 - 2 using the following criteria:
    2: Both the literal and contextual meanings are similar in both the source text and the translation
    1: The literal meaning of the pun’s wordplay is similar in both the source text and the translation, but the translation fails to convey the contextual meaning 
        or intended humor of the original pun.
    0: The pun’s wordplay is mistranslated, meaning that both the literal and contextual meanings differ between the source and translation, resulting in a complete
        loss of the intended pun or humor.

Evaluate the pun based on these criteria and provide a justification in english for your rating. Provide your answer in the following format:

Rating: <number from 0 - 2>
Justification: <very concise explanation of key issues>"""
    },
    "emotion": {
        "instructions": """You are fluent in both English and French, with a deep understanding of humor in both languages. Your task is to compare a translated pun to 
        its original version and assess to what extent the original pun's wordplay and its translation convey different amounts of emotion.

Rate the emotion of the translation compared to the original using 0 or 1:
    0 : Less
    0 : More
    1 : Same

Evaluate the pun based on these criteria and provide a justification in english for your rating. Provide your answer in the following format:

Rating: <number from 0 - 1>
Justification: <very concise explanation of key issues>"""
    },
    "authenticity": {
        "instructions": """You are fluent in both English and French, with a deep understanding of humor in both languages. Your task is to compare a translated pun to 
        its original version and assess to what extent the translated pun reads like standard, well-edited language, such that the pun would be understood by a native speaker of 
        the French language.

Rate the translation 0 - 4 using one of the following categories:
    0 : Not at all likely
    1 : Not Very Likely
    2 : Somewhat Likely
    3 : Very Likely
    4 : Extremely Likely

Evaluate the pun based on these criteria and provide a justification in english for your rating. Provide your answer in the following format:

Rating: <number from 0 - 4>
Justification: <very concise explanation of key issues>"""
    },
}


def parse_evaluator_response(text):
    # Try to extract rating
    rating_match = re.search(r"Rating:\s*(\d+)", text)
    rating = int(rating_match.group(1)) if rating_match else None

    # Extract justification
    justification_match = re.search(r"Justification:\s*(.*)", text, re.DOTALL)
    justification = (
        justification_match.group(1).strip()
        if justification_match
        else "No justification"
    )

    return {"rating": rating, "justification": justification}


def aggregate_evaluations(evaluator_responses):
    thresholds = {
        "equivalence": 2,
        "mistranslation": 2,
        "emotion": 1,
        "authenticity": 3,
    }

    # Parse responses
    evaluations = {
        key: parse_evaluator_response(resp) for key, resp in evaluator_responses.items()
    }

    # Extract scores
    scores = {
        key: eval["rating"]
        for key, eval in evaluations.items()
        if eval["rating"] is not None
    }

    # Check if we received scores from all evaluators
    if len(scores) < len(thresholds):
        print("Missing scores from some evaluators. Re-evaluation.")
        return "refine"

    # Determine if all scores meet thresholds
    if all(scores[key] >= thresholds[key] for key in thresholds):
        print("Translation is good enough. Accepting it.")
        return "accept"

    print("Translation needs improvement.")
    return "refine"


# The input DataFrame should have the following columns:
# - "text_clean": The original English pun text.
# - "initial_translation": The initial French translation of the pun.
# - "id_en": An identifier for the English pun.
# - "is_pun": A binary indicator (1 for pun, 0 for non-pun).
# - change the dataframe this is loading from in config.ini at identification_gpt_4o_path
# - This saves to refinement_gpt_4o_path
# - Sentences identified as non puns and were not refined will have a "0" in the "iteration" column.


def refine_translations(df, model):
    for idx, row in df.iterrows():
        english_pun = row["text_clean"]
        current_translation = row["initial_translation"]
        id_en = row["id_en"]
        is_pun = row["is_pun"]
        max_iterations = 5
        iteration = 0
        if is_pun == 1:
            iteration = 1
            while iteration < max_iterations + 1:
                print(
                    f"\n--------------Iteration {iteration} for Row {idx}---------------"
                )
                print("\nCURRENT TRANSLATION: ", current_translation)

                evaluator_responses = {}
                for key, evaluator in evaluators.items():
                    input_text = f"{evaluator['instructions']}\n\n{english_pun}\n\n{current_translation}"
                    response = get_response_not_json(input_text, model)
                    evaluator_responses[key] = response

                decision = aggregate_evaluations(evaluator_responses)

                if decision in ["accept", "minor_fix"]:
                    print(
                        f"Final translation after {iteration} iteration(s): {current_translation}"
                    )

                    df.at[idx, "id_en"] = id_en
                    df.at[idx, "final_translation"] = current_translation
                    df.at[idx, "iteration"] = iteration
                    break

                feedback_text = "\n\n".join(
                    [
                        f"{key.upper()} FEEDBACK:\n{resp}"
                        for key, resp in evaluator_responses.items()
                    ]
                )

                refinement_prompt = f"""
                The following pun was translated into French but did not meet all quality standards.
                
                English Pun: {english_pun}
                Current French Translation: {current_translation}
                
                Here is feedback from evaluators:
                {feedback_text}
                
                Improve the translation based on this feedback.
                
                Provide an improved translation. Only provide the French translation in your response:
                """

                current_translation = get_response_not_json(refinement_prompt, model)
                iteration += 1

            if iteration == max_iterations:
                print(
                    "Max refinements reached. Final translation:", current_translation
                )
                df.at[idx, "id_en"] = id_en
                df.at[idx, "final_translation"] = current_translation
                df.at[idx, "iteration"] = iteration

    return df


if __name__ == "__main__":
    model = "gpt-4o"

    df = load(identification_gpt_4o_path)
    df = refine_translations(df, model)
    save(df, refinement_gpt_4o_path)
