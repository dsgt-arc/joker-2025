# overview

1. scan the source text and flag possible puns
	1. prompt to find pun_word	

	output: text_en, pun_word

2. identify the incongruous meanings using word sense and semantic role knowledge-bases,
	1. prompt whether homographic or homophonic
	
	2. get alternative meaning
		if homographic:
			prompt to find alternative_meaning
			prompt to get list of synonyms of pun_word that do not mean alternative_meaning
			prompt to get list of synonyms of alternative_meaning that do not mean pun_word
			
		if homophonic:
			prompt to find alternative_word
			optional [prompt do pun_word and alternative_word mean the same thing?
				if yes, try again (loop)]
			prompt to get list of synonyms of pun_word [that do not mean alternative_word]
			prompt to get list of synonyms of alternative_word [that do not mean pun_word]

		output: text_en, pun_type, pun_synonyms, alternative_synonyms

3. look up translations of the pun’s two meanings and search for closely related senses in the target language
	1. prompt to translate pun_synonyms, alternative_synonyms

	prompt to get list of French synonyms of pun_synonyms_fr
	prompt to get list of French synonyms of alternative_synonyms_fr
	combine the lists

	2. check for homonyms
		for each pun_synonym_fr
			prompt is pun_synonym_fr a homonym?
			for each alternative_synonym_fr
				prompt does pun_synonym_fr overlap alternative_synonym_fr

			if homonym and overlaps:
				direct translation true
				direct_translation_pun_word
				direct_translation_alternative
			else:
				direct_translation false

	output: text_en, pun_synonyms_fr, alternative_synonyms_fr

4. search among those results to find phonetically similar candidates

5. repeat the above to generate a set of candidate translations

6. rank the candidate translations and select the most promising
