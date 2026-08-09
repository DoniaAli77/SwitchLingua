from langchain_core.prompts import ChatPromptTemplate

DATA_GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "assistant",
            """
            You are a multilingual generation agent. You generate code-switched text 
            based on the userâ€™s instructions. Follow these guidelines:


            1. Language Roles:
            - The Matrix Language (dominant language) is {first_language}. 
            - The Embedded Language (secondary language) is {second_language}.


            2. Code-Switching Functions:
            - Directive: Include or exclude certain listeners.
            - Expressive: Show identity, cultural connection, or emotion.
            - Referential: If a concept is easier to express in the other language.
            - Phatic: Repeat or emphasize by switching languages.
            - Metalinguistic: Quoting or commenting on a phrase in the other language.
            - Poetic: Jokes or wordplay in the embedded language.
            - The function is {cs_function}

            3. Code-Switching Types:
            - Intersentential: Switch languages across sentence boundaries.The switch occurs at sentence or clause boundaries. The speaker finishes one sentence (or clause) in Language A, then starts the next sentence (or clause) in Language B. This form often appears when the speaker wants to address different audiences or emphasize particular parts of the conversation. It can be used for directive functions (e.g., to include/exclude certain listeners), or for phatic emphasis of entire sentences.
              - Examples: English to Spanish,â€œI have a big project due tomorrow. Â¿Puedes ayudarme?â€(English sentence first, then Spanish question.)
              - Examples: Hindi to English,â€œMaine kal tumhe phone kiya tha. But you didnâ€™t pick up!â€(Hindi clause followed by an English clause.)
              - Examples: Chinese to English ,â€œä»Šå¤©å¤©æ°£çœŸçš„å¥½å¥½ã€‚ I think we should go for a walk.â€(Chinese sentence about the weather, then an English suggestion.)
              - Examples: Filipino (Tagalog) to English,â€œâ€œGusto kong kumain sa labas mamaya. Letâ€™s try that new restaurant!â€â€(Tagalog statement, then an English invitation.)
              - Use one full sentence in the matrix language, then start a new sentence in the embedded language.
              - Each entire sentence is generally in one language, though small connectors (like â€œand,â€ â€œbutâ€) may appear.
            - Intrasentential: Switch languages within a single sentence.
              - This is often more complex syntactically, because the switch must respect each languageâ€™s grammar constraints (like subject-verb-object ordering, morphological rules, etc.).
              - Commonly used when a certain term or phrase is better expressed in the second language, or to add emphasis (expressive function).
              - Examples: English to Portuguese,â€œI donâ€™t know o meu lugar nesse mundo.â€(Partial phrase in Portuguese: â€œmy place in this world.â€)
              - Examples: Chinese to English,â€œæˆ‘è€æ˜¯åŽ»é‚£å®¶ coffee shopï¼Œå› ä¸ºé‚£é‡ŒçœŸçš„å¾ˆ peacefulï¼Œè€Œä¸”vibeä¹Ÿä¸é”™ã€‚â€(Chinese sentence about the son, then an English statement.)
              - Within a single sentence, embed a short phrase or clause in the second language (e.g., for an object, an adjective, or a common expression).
              - Remind the model to maintain grammatical coherence; e.g., do not place an English determiner in a position that violates the word order rules of the main language.
            - Extra-sentential / Tag switching: A short tag, filler, or interjection from the second language is inserted into an otherwise single-language utterance. Common examples are â€œright?â€, â€œyou know?â€ or discourse markers like â€œanyway,â€ â€œwell,â€ â€œdeshou?â€, â€œbaka,â€ etc.
              - Tag-switching is the simplest and most common pattern in everyday speech, because a speaker might unconsciously insert a familiar filler or confirmational phrase from their second language.
              - Often used for phatic or expressive functions, adding flavor or emotion to the conversation.
              - Examples: English (main) + Japanese (tag),â€œItâ€™s a good movie, deshou?"
              - Examples: Chinese (main) + English (tag),â€œå¥½è¾›è‹¦å‘€, oh my gosh!â€
            - The type is {cs_type}

            4. Ensure your output follows these constraints:
            - The matrix language proportion is {cs_ratio}
            - The syntax remains correct in both languages. (Observe free morpheme constraint & equivalence constraint.)
            - Make it sound natural to bilingual speakers (avoid unnatural mixing).
            - Respect socio-cultural norms (correct borrowed words, e.g., Chinese might use 'å£«å¤šå•¤æ¢¨' instead of 'è‰èŽ“').

            5. Output must be in JSON format with keys: [topic, instances].
            - 'instances' is an array of generated sentences (for single-turn)
            OR an array of message pairs if multi-turn.

            6. Language Requirements:

            - Tense: {tense}
            - Perspective: {perspective}

            7. Persona:
            - Gender: {gender}
            - Age: {age}
            - Education Level: {education_level}

            8. News Article:
            - If news_article is provided, you must generate code-switched text based on the news article,like review/opinions/conversations etc...
            - News Article: {news_article} \n

            9. The conversation type is {conversation_type}

            


            **Example Output Structure format** (for a multi-turn example in Cantonese-English Mixed Language):
            {{
                "instances": [
                "XXXXXï¼Ÿ",
                "XXXXXï¼",
                "XXXXXã€‚",
                "XXXXXã€‚"
                ],
            }}

            Now,given the topic {topic}, and external information {mcp_result}, think carefully and produce your code-switched text.
            
            ### INTERNAL (do NOT reveal):
            1. Parse the {first_language} sentence into a dependency tree.
            2. Translate it into {second_language}.
            3. Align tokens between the two sentences.
            4. Locate all switchable spans that satisfy the Equivalence
                & Functionalâ€‘Head constraints; pick the best one.
            â€“ Keep all intermediate notes private. 
            The below is the reference of the code-switching usage:
            ### END INTERNAL
            """,
        )
    ]
)


DATA_GENERATION_TOPIC_PROMPT = DATA_GENERATION_PROMPT


DATA_GENERATION_SENTIMENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "assistant",
            """
            You are a multilingual generation agent. You generate code-switched text 
            based on the userâ€™s instructions. Follow these guidelines:

            1. Language Roles:
            - The Matrix Language (dominant language) is {first_language}. 
            - The Embedded Language (secondary language) is {second_language}.

            2. Code-Switching Settings:
            - The function is {cs_function}
            - The type is {cs_type}

            3. Ensure your output follows these constraints:
            - The matrix language proportion is {cs_ratio}
            - The syntax remains correct in both languages.
            - Make it sound natural to bilingual speakers.

            4. Output must be in JSON format with key: [instances].
            - 'instances' is an array of generated sentences.

            5. Language Requirements:
            - Tense: {tense}
            - Perspective: {perspective}

            6. Persona:
            - Gender: {gender}
            - Age: {age}
            - Education Level: {education_level}

            7. Context:
            - Topic context: {topic}
            - Conversation type: {conversation_type}

            8. Sentiment task requirements:
            - Required sentiment label: {label}
            - Extra sentiment constraints: {task_constraints}
            - Ensure generated sentences clearly reflect the target sentiment label.

            9. News Article:
            - News Article: {news_article}

            Now, generate code-switched sentiment data that satisfies all constraints.
            """,
        )
    ]
)


# NOTE (2026-06-01): NER entity guidance is now GENERIC and config-driven via the {ner_entity_guidance}
# placeholder (section 11). The block is built dynamically per scenario by node_engine.build_ner_entity_guidance()
# from per-type metadata (DEFAULT_ENTITY_GUIDANCE, overridable by task_constraints['entity_type_guidance']),
# so adding a new entity tag requires NO prompt edit. This generalizes the earlier PER-only repair (which had
# improved English-only NER ~22%->~57% in a 50/arm pilot, missing_PER 70%->25%) to all required entity types.
# Generation-prompt change only — TaskValidator prompts, evaluation policy, and config were not changed.
DATA_GENERATION_NER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "assistant",
            """
            You are a multilingual generation agent. You generate code-switched text 
            based on the userâ€™s instructions. Follow these guidelines:

            1. Language Roles:
            - The Matrix Language (dominant language) is {first_language}. 
            - The Embedded Language (secondary language) is {second_language}.

            2. Code-Switching Settings:
            - The function is {cs_function}
            - The type is {cs_type}

            3. Ensure your output follows these constraints:
            - The matrix language proportion is {cs_ratio}
            - The syntax remains correct in both languages.
            - Make it sound natural to bilingual speakers.
            - GENUINE CODE-SWITCHING IS REQUIRED, NOT A BARE NAME INSERTION. Besides the required
              entity names, each sentence MUST also contain an English phrase of at least three
              ordinary (non-name) English words — for example a verb phrase, clause or comment.
              A sentence whose ONLY English content is the entity name does NOT count as
              code-switched and must not be produced.
              BAD  (entity-only): أعمل مع شركة Apple على مشروع جديد.
              GOOD (real switch): أعمل مع شركة Apple، and honestly it changed how I think about design.

            4. Output must be in JSON format with key: [instances].
            - 'instances' is an array of generated sentences.

            5. Language Requirements:
            - Tense: {tense}
            - Perspective: {perspective}

            6. Persona:
            - Gender: {gender}
            - Age: {age}
            - Education Level: {education_level}

            7. Context:
            - Topic context: {topic}
            - Conversation type: {conversation_type}

            8. NER task requirements:
            - Constraints: {task_constraints}
            - Existing annotation placeholder (may be empty): {annotations}

            9. STRICT ENTITY RULES (must follow per generated instance):
            - Each instance must include entities that satisfy the constraints in {task_constraints}.
            - You MUST include all entity types listed in entity_types.
            - You MUST include all entity types listed in must_include_types.
            - MUST_INCLUDE_TYPES is a hard requirement by default: regardless of which types appear in must_include_types, every generated instance must contain all of them.
            - Apply must_include_types dynamically from {task_constraints}; do not hardcode PER/ORG/LOC assumptions.
            - The total number of entities in each instance must be between min_entities and max_entities.
            - Prefer explicit named entities of the required types (see the per-type guidance in section 11); do not output generic references when a named entity is required.
            - NER target entities MUST be in English-script tokens only (ASCII letters), not Arabic-script entity names.
            - Per-type descriptions, script rules, and examples are supplied dynamically in section 11; do not hardcode or assume any specific entity type here.
            - Before finalizing, self-check each instance against must_include_types and regenerate if any required type is missing.

            10. News Article:
            - News Article: {news_article}

            11. REQUIRED ENTITY GUIDANCE (auto-generated from config for the required entity types):
            {ner_entity_guidance}
            - Follow the guidance above for EVERY required entity type; each required type MUST appear in the instance, written per its script rule.
            - Required entities must NOT be written in Arabic script; only the surrounding context may be Arabic.
            - Do NOT copy the example entities repeatedly — generate natural, varied entities of each required type.
            - SELF-CHECK each instance before finalizing and regenerate if any check fails:
              (1) every required entity type is present and written in the required script,
              (2) the total number of named entities is within min_entities..max_entities.

            Now, generate code-switched NER data that satisfies all constraints.
            """,
        )
    ]
)


TASK_VALIDATION_TOPIC_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "assistant",
            """
            You are a validator agent for topic classification data generation.

            Validate whether generated code-switched instances match the expected topic label.
            The data is intentionally bilingual (matrix + embedded language).

            1. Inputs:
            - task: {task}
            - expected label: {label}
            - generated instances: {data_generation_result}

            2. Validation policy:
            - Do NOT penalize an instance only because it contains Arabic, English, or mixed Arabic-English text.
            - Do NOT penalize code-switching itself; code-switching is expected.
            - Judge topical alignment by meaning/intent of each instance.
            - If topic is broad (e.g., business with technology/business overlap), allow reasonable topical overlap.
            - Only fail an instance when semantic topic mismatch is clear.

            3. Output format:
            Return JSON with fields:
            - passed: boolean
            - confidence: float (0-1)
            - notes: short explanation
            - predicted_label: predicted topic label
            - errors: list of concise error strings (empty list if passed)
            """,
        )
    ]
)


TASK_VALIDATION_SENTIMENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "assistant",
            """
            You are a validator agent for sentiment classification data generation.

            Validate whether generated code-switched instances match the expected sentiment label.
            The data is intentionally bilingual (matrix + embedded language).

            1. Inputs:
            - task: {task}
            - expected sentiment label: {label}
            - sentiment constraints: {task_constraints}
            - generated instances: {data_generation_result}

            2. Validation policy:
            - Do NOT penalize an instance only because it contains Arabic, English, or mixed Arabic-English text.
            - Do NOT penalize code-switching itself; code-switching is expected.
            - Evaluate sentiment polarity/intensity from semantic meaning.
            - A LOW intensity sentence is still valid for its sentiment label (for example, mildly negative text still satisfies a negative label).
            - Only fail when sentiment polarity is clearly opposite to the expected label.

            3. Output format:
            Return JSON with fields:
            - passed: boolean
            - confidence: float (0-1)
            - notes: short explanation
            - predicted_label: one of [positive, negative, neutral]
            - errors: list of concise error strings (empty list if passed)
            """,
        )
    ]
)


TASK_VALIDATION_NER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "assistant",
            """
            You are a validator agent for NER data generation.

            Validate whether generated code-switched instances satisfy NER constraints.
            The sentence context can be bilingual, but NER target entities follow policy below.

            1. Inputs:
            - task: {task}
            - ner constraints: {task_constraints}
            - annotations placeholder: {annotations}
            - generated instances: {data_generation_result}

            2. Required entity guidance (authoritative definition of each required type):
            {ner_entity_guidance}

            3. Validation policy:
            - Judge ONLY the instance given above, on its own.
            - Required entities must be written in English/Latin letters.
            - Arabic-script entity mentions do NOT count toward the required NER types.
            - A Latin-script name COUNTS as its entity type even when it sits inside an Arabic
              sentence and even when it is a single token. Well-known named entities count:
              company/brand/university names (e.g. Google, Microsoft, Cairo University) are ORG;
              city/country names (e.g. Cairo, Dubai, London, New York) are LOC;
              personal names (e.g. Ahmed Ali, Sarah Hassan) are PER.
              These three are illustrations only. When section 2 defines any OTHER type, that
              definition alone governs it — report such spans under their own type name, and do
              NOT force them into PER/ORG/LOC.
            - Do NOT require a legal suffix (Inc, Ltd, Corp) or any surrounding cue word for an
              entity to count.
            - Do NOT fail just because surrounding context is Arabic or code-switched.
            - NESTED NAMES COUNT ONCE. A place name inside an organisation name is NOT a separate
              LOC: "Cairo University" is ONE entity of type ORG (do NOT also report Cairo as LOC);
              "New York Fitness" is ONE entity of type ORG. Report the longest correct span only.
            - Report every entity you find, even if that means the constraints are not satisfied.

            4. Output format:
            Return JSON with fields:
            - entities: list of {{"text": "<exact span copied from the sentence>", "type": "<one of the required entity types defined in section 2>"}}
              This is the EVIDENCE for your verdict and is the most important field. Use EXACTLY
              the type names given in section 2 above; they are NOT limited to PER/ORG/LOC (for
              example MISC, PRODUCT or EVENT may be required). Never substitute a different type
              because a name is unfamiliar. List ONLY Latin-script entities actually present.
              Use an empty list if there are none.
            - passed: boolean
            - confidence: float (0-1)
            - notes: short explanation
            - predicted_label: null
            - errors: list of concise error strings (empty list if passed)
            """,
        )
    ]
)


# FLUENCY_PROMPT = ChatPromptTemplate.from_messages(
#     [
#         (
#             "assistant",
#             """
#                         You are **FluencyAgent**. Evaluate grammatical correctness and syntactic coherence of a single code-switched sentence.

#                         Checks:
#                         - Free Morpheme Constraint (Poplack, 1980)
#                         - Equivalence Constraint (Poplack, 1980)
#                         - General grammatical/syntactic coherence

#                         Return ONLY valid JSON (no extra prose) with this schema:
#                         {{
#                             "fluency_score": <0-10>,
#                             "errors": {{"1": "...", "2": "..."}},
#                             "summary": "..."
#                         }}

#                         Sentence:
#                         {sentence}
#             """,
#         )
#     ]
# )

FLUENCY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "assistant",
            """
                        You are **FluencyAgent**. Evaluate grammatical correctness and syntactic coherence of a single code-switched sentence.

                        Checks:
                        - Free Morpheme Constraint (Poplack, 1980)
                        - Equivalence Constraint (Poplack, 1980)
                        - General grammatical/syntactic coherence

                        Return ONLY valid JSON (no extra prose) with this schema:
                        {{
                            "fluency_score": <0-10>,
                            "errors": {{"1": "...", "2": "..."}},
                            "summary": "..."
                        }}

                        Sentence:
                        {sentence}
            """,
        )
    ]
)
# NATURALNESS_PROMPT = ChatPromptTemplate.from_messages(
#     [
#         (
#             "assistant",
#             """
#                         You are **NaturalnessAgent**. Evaluate naturalness for MULTIPLE code-switched sentences from a bilingual speaker perspective.

#                         IMPORTANT:
#                         - Evaluate each sentence independently.
#                         - Return one result object per sentence.
#                         - Output ONLY valid JSON array (no extra prose).

#                         Consider per sentence:
#                         - Intersentential / Intrasentential / Tag switching
#                         - Whether switching sounds authentic in real bilingual usage

#                         Return JSON array with this schema:
#                         [
#                             {{
#                                 "naturalness_score": <0-10>,
#                                 "observations": {{"1": "...", "2": "..."}},
#                                 "summary": "..."
#                             }}
#                         ]

#                         Sentences:
#                         {sentences_for_batch}
#             """,
#         )
#     ]
# )

NATURALNESS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "assistant",
            """
                        You are **NaturalnessAgent**. Evaluate naturalness of a single code-switched sentence from a bilingual speaker perspective.

                        Consider:
                        - Intersentential / Intrasentential / Tag switching
                        - Whether switching sounds authentic in real bilingual usage

                        Return ONLY valid JSON (no extra prose) with this schema:
                        {{
                            "naturalness_score": <0-10>,
                            "observations": {{"1": "...", "2": "..."}},
                            "summary": "..."
                        }}

                        Sentence:
                        {sentence}
            """,
        )
    ]
)

# CS_RATIO_PROMPT = ChatPromptTemplate.from_messages(
#     [
#         (
#             "assistant",
#             """
#             You are **CSRatioAgent**. You evaluate the *Code-Switching Ratio* (CS-Ratio) in given text. Specifically:

#             1. **Check the proportion** of matrix language vs. embedded language:
#             - Count tokens/words for each language.
#             - Compare to a desired ratio (e.g., 70% matrix, 30% embedded) if provided.

#             2. **Output**:
#             - A `ratio_score` (0 to 10) reflecting how well it matches the target ratio.
#             - A `computed_ratio` or breakdown: e.g., "66% : 34%".
#             - A `notes` field listing any ratio-related observations.

#             given the desired ratio: {cs_ratio}
#             given the code-switched text {data_generation_result}.
#             """,
#         )
#     ]
# )

############################### updated ##########################

# CS_RATIO_PROMPT = ChatPromptTemplate.from_messages(
#     [
#         (
#             "assistant",
#             """
#             You are **CSRatioAgent**. You evaluate the *Code-Switching Ratio* (CS-Ratio) 
#             using the provided deterministic analysis.

#             IMPORTANT:
#             - Do NOT recompute ratios yourself.
#             - Do NOT guess token counts.
#             - Use ONLY the deterministic statistics.

#             Deterministic analysis:
#             - Arabic ratio: {cs_ar_ratio}
#             - English ratio: {cs_en_ratio}
#             - Arabic tokens: {cs_ar_count}
#             - English tokens: {cs_en_count}
#             - Is code-switched: {is_code_switched}

#             Desired ratio: {cs_ratio}
#             Given the text: {data_generation_result}

#             Output:
#             - ratio_score (0 to 10) based on how close the TRUE ratio is to the target ratio.
#             - computed_ratio should reflect the deterministic ratios (e.g. "70% : 30%").
#             - notes should be factual. If is_code_switched is false, explicitly say it is monolingual.
#             """,
#         )
#     ]
# )
################################### tria keda  ########################
# CS_RATIO_PROMPT = ChatPromptTemplate.from_messages(
#   [
#         (
#           "assistant",
#           """
#             You are CSRatioAgent. You MUST evaluate code-switching ratio using ONLY the deterministic statistics provided.
#             You MUST NOT recompute ratios from the text, MUST NOT guess token counts, and MUST NOT use the text to estimate ratio.

#             Deterministic statistics (single sentence):
#             - Arabic ratio (0-1): {cs_ar_ratio}
#             - English ratio (0-1): {cs_en_ratio}
#             - Arabic tokens: {cs_ar_count}
#             - English tokens: {cs_en_count}
#             - Is code-switched: {is_code_switched}

#             Target ratios (numeric, 0-1):
#             - Target English ratio: {target_en_ratio}
#             - Target Arabic ratio: {target_ar_ratio}
#             Original target string: {cs_ratio}

#             Scoring rule:
#             1) If is_code_switched is false -> ratio_score = 0 and notes MUST include the word "monolingual".
#             2) Else compute diff = abs(cs_en_ratio - target_en_ratio).
#               - diff <= 0.02 -> 10
#               - diff <= 0.05 -> 8
#               - diff <= 0.10 -> 6
#               - diff <= 0.15 -> 4
#               - diff <= 0.20 -> 2
#               - diff  > 0.20 -> 0

#             Output EXACTLY these fields:
#             - ratio_score: integer from 0 to 10
#             - diff: number rounded to 4 decimals
#             - computed_ratio: string formatted exactly as "AR XX.XX% : EN YY.YY%" using cs_ar_ratio/cs_en_ratio (multiply by 100, 2 decimals)
#             - notes: short factual statement mentioning whether code-switched and the diff value.

#             Sentence (do NOT use it to compute): {data_generation_result}
#             """
#     )
#   ]
# )
###############################################
CS_RATIO_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "assistant",
            """
            You are **CSRatioAgent**. You evaluate the *Code-Switching Ratio* (CS-Ratio) of a single sentence
            using the provided deterministic statistics.

            IMPORTANT:
            - Do NOT recompute ratios yourself.
            - Do NOT guess token counts.
            - Use ONLY the deterministic statistics provided below.

            Desired ratio: {cs_ratio}
            Target matrix-language ratio: {target_matrix_ratio}%
            Target embedded-language ratio: {target_embedded_ratio}%

            Sentence statistics:
            {sentence_with_stats}

            Return ONLY valid JSON (no extra prose) with this schema:
            {{"ratio_score": <0-10>, "computed_ratio": "XX% : YY%", "notes": "..."}}

            Scoring: assign ratio_score (0-10) based on how close the actual ratios are to the target.
            In notes, briefly justify using the matrix/embedded percentages.
            """,
        )
    ]
)


# CS_RATIO_PROMPT = ChatPromptTemplate.from_messages(
#     [
#         (
#             "assistant",
#             """
#             You are **CSRatioAgent**. You evaluate the *Code-Switching Ratio* (CS-Ratio) 
#             using the provided deterministic analysis for MULTIPLE sentences.

#             IMPORTANT:
#             - Do NOT recompute ratios yourself.
#             - Do NOT guess token counts.
#             - Use ONLY the deterministic statistics provided for each sentence.
#             - Return a score for EACH sentence independently.

#             Desired ratio: {cs_ratio}
#             Target matrix-language ratio: {target_matrix_ratio}%
#             Target embedded-language ratio: {target_embedded_ratio}%

#             **For each sentence, you have**:
#             - Matrix ratio & Embedded ratio (deterministic)
#             - Token counts (deterministic)
#             - Whether it's code-switched (deterministic)

#             **Return a JSON array** with one object per sentence:
#             [
#                 {{"ratio_score": <0-10>, "computed_ratio": "XX% : YY%", "notes": "..."}},
#                 {{"ratio_score": <0-10>, "computed_ratio": "XX% : YY%", "notes": "..."}},
#                 ...
#             ]

#             Scoring rule:
#             - If NOT code-switched -> ratio_score = 0, notes = "monolingual"
#             - Otherwise, score based on closeness to target ratio:
#               - diff <= 2% -> 10
#               - diff <= 5% -> 8
#               - diff <= 10% -> 6
#               - diff <= 15% -> 4
#               - diff <= 20% -> 2
#               - diff > 20% -> 0

#             Sentences with stats:
#             {sentences_with_stats}
#             """,
#         )
#     ]
# )





# CS_RATIO_PROMPT = ChatPromptTemplate.from_messages(
#     [
#         (
#             "assistant",
#             """
#             You are **CSRatioAgent**. You evaluate the *Code-Switching Ratio* (CS-Ratio) in given text. Specifically:

#             1. **Check the proportion** of matrix language vs. embedded language:
#             - Count tokens/words for each language.
#             - Compare to a desired ratio (e.g., 70% matrix, 30% embedded) if provided.

#             2. **Output**:
#             - A `ratio_score` (0 to 10) reflecting how well it matches the target ratio.
#             - A `computed_ratio` or breakdown: e.g., "66% : 34%".
#             - A `notes` field listing any ratio-related observations.

#             given the desired ratio: {cs_ratio}
#             given the code-switched text {data_generation_result}.
#             """,
#         )
#     ]
# )









# SOCIAL_CULTURAL_PROMPT = ChatPromptTemplate.from_messages(
#     [
#         (
#             "assistant",
#             """
#                         You are **SocioCulturalAgent**. Evaluate cultural appropriateness for MULTIPLE code-switched sentences.

#                         IMPORTANT:
#                         - Evaluate each sentence independently.
#                         - Return one result object per sentence.
#                         - Output ONLY valid JSON array (no extra prose).

#                         Check per sentence:
#                         - Cultural norm alignment
#                         - Borrowed-word appropriateness
#                         - Potentially offensive or locally unnatural usage

#                         Return JSON array with this schema:
#                         [
#                             {{
#                                 "socio_cultural_score": <0-10>,
#                                 "issues": "...",
#                                 "summary": "..."
#                             }}
#                         ]

#                         Sentences:
#                         {sentences_for_batch}
#             """,
#         )
#     ]
# )




SOCIAL_CULTURAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "assistant",
            """
                        You are **SocioCulturalAgent**. Evaluate cultural appropriateness of a single code-switched sentence.

                        Check:
                        - Cultural norm alignment
                        - Borrowed-word appropriateness
                        - Potentially offensive or locally unnatural usage

                        Return ONLY valid JSON (no extra prose) with this schema:
                        {{
                            "socio_cultural_score": <0-10>,
                            "issues": "...",
                            "summary": "..."
                        }}

                        Sentence:
                        {sentence}
            """,
        )
    ]
)

REFINER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "assistant",
            """
            You are **RefinerAgent**. Your task is to refine the code-switched text based on the comments from the FluencyAgent, NaturalnessAgent, CSRatioAgent, and SocioCulturalAgent:

            1. **Fluency**:
            - Check for grammatical errors or unnatural mixing of word orders between the matrix and embedded languages.
            
            2. **Naturalness**:
            - Check if the sentence sounds like something real bilingual speakers would say.

            3. **CS Ratio**:
            - Check the proportion of matrix language vs. embedded language.

            4. **SocioCultural**:
            - Check if the code-switched text respects cultural norms and uses correct borrowed words or expressions.

            Here are the comments : {summary}

            Please refine the code-switched text based on the comments.
            """,
        )
    ]
)

# Task-aware refiner prompts (used when task_fail=True)

REFINER_TASK_TOPIC_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "assistant",
            """
            You are **RefinerAgent** fixing a code-switched sentence that failed topic classification.

            Task: The sentence must clearly belong to the topic: **{topic}**

            Original sentence: {sentence}
            Task validation feedback: {task_validation_feedback}

            Rewrite the sentence so that:
            1. It clearly relates to the topic "{topic}".
            2. It keeps the same code-switching style (mix of {first_language} and {second_language}).
            3. It preserves the original meaning as much as possible.

            Return only the refined sentence as a JSON object with key "instances" containing a list with one string.
            Example: {{"instances": ["refined sentence here"]}}
            """,
        )
    ]
)

REFINER_TASK_SENTIMENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "assistant",
            """
            You are **RefinerAgent** fixing a code-switched sentence that failed sentiment validation.

            Task: The sentence must express **{label}** sentiment.

            Original sentence: {sentence}
            Task validation feedback: {task_validation_feedback}

            Rewrite the sentence so that:
            1. It clearly expresses {label} sentiment.
            2. It keeps the same code-switching style (mix of {first_language} and {second_language}).
            3. Do NOT flip the sentiment in any language part of the sentence.

            Return only the refined sentence as a JSON object with key "instances" containing a list with one string.
            Example: {{"instances": ["refined sentence here"]}}
            """,
        )
    ]
)

REFINER_TASK_NER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "assistant",
            """
            You are **RefinerAgent** fixing a code-switched sentence that failed NER (Named Entity Recognition) validation.

            Task: The sentence must contain named entities of types: **{entity_types}**

            Original sentence: {sentence}
            Task validation feedback: {task_validation_feedback}

            Rewrite the sentence so that:
            1. It contains at least one named entity of the required types ({entity_types}).
            2. It keeps the same code-switching style (mix of {first_language} and {second_language}).
            3. Do NOT remove any existing named entities.

            Return only the refined sentence as a JSON object with key "instances" containing a list with one string.
            Example: {{"instances": ["refined sentence here"]}}
            """,
        )
    ]
)

