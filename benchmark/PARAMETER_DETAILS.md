## Evaluating a Quantized Model Using Various Benchmarks

The provided code snippet illustrates how to evaluate a quantized model with different benchmark datasets using Python. Below is a detailed explanation of each function interface and the parameters they accept:

### 1. `Benchmark.eval_ppl(model, tokenizer, test_dataset)`
- **Parameters:**
  - `model` (object): The quantized model loaded using `AutoModelForCausalLM`.
  - `tokenizer` (object): The tokenizer loaded using `AutoTokenizer`.
  - `test_dataset` (list of str): List of dataset identifiers to evaluate the model's perplexity. Options include `['wikitext2', 'c4', 'ptb']`.

### 2. `Benchmark.eval_ceval(model, tokenizer, model_type, subject, num_shot)`
- **Parameters:**
  - `model` (object): The quantized model.
  - `tokenizer` (object): The tokenizer.
  - `model_type` (str): Type of the model to evaluate. Options are `['chatglm', 'baichuan', 'llama']`.
  - `subject` (str): Subject area for evaluation. Options are `['all', 'STEM', 'Social Science, 'Humanities', 'Other']`.
  - `num_shot` (int): Number of examples used for few-shot learning. Range is 0 to 5.

### 3. `Benchmark.eval_mmlu(model, tokenizer, model_type, subject, num_shot)`
- **Parameters:**
  - Similar to `eval_ceval`, but the `subject` options differ slightly, including `['all', 'STEM', 'Social Sciences', 'Humanities', 'China Specific', 'Other']`.

### 4. `Benchmark.eval_boss(model, tokenizer, test_dataset, split, ICL_split, num_shot)`
- **Parameters:**
  - `model` (object): The quantized model.
  - `tokenizer` (object): The tokenizer.
  - `test_dataset` (str): Dataset to evaluate. Options are 
  - `QuestionAnswering_squad`
  - `QuestionAnswering_advqa`
  - `QuestionAnswering_newsqa`
  - `QuestionAnswering_searchqa`
  - `SentimentAnalysis_amazon`
  - `SentimentAnalysis_dynasent`
  - `SentimentAnalysis_semeval`
  - `SentimentAnalysis_sst5`
  - `NaturalLanguageInference_mnli`
  - `NaturalLanguageInference_anli`
  - `NaturalLanguageInference_wanli`
  - `NaturalLanguageInference_contractnli`
  - `ToxicDetection_civilcomments`
  - `ToxicDetection_advcivil`
  - `ToxicDetection_implicithate`
  - `ToxicDetection_toxigen`
  - `split` (str): Dataset split to evaluate, generally set to 'test'.
  - `ICL_split` (str): In-context learning split, usually set to 'test'.
  - `num_shot` (int): Number of examples for few-shot learning, typically 0.

### 5. `Benchmark.eval_lmeval(model, eval_tasks, num_shot)`
- **Parameters:**
  - `model` (object): The quantized model.
  - `tokenizer` (object): The tokenizer.
  - `eval_tasks` (list of str): List of tasks for evaluation. Includes `["winogrande", "piqa", "hellaswag"]`.
  - `num_shot` (int): Number of examples for few-shot learning, options are 0 or 5.

Each function is designed to test different aspects of the model's capabilities using a variety of datasets and tasks, enabling comprehensive performance assessments.

## `eval_tasks` Options for `Benchmark.eval_lmeval`

The `Benchmark.eval_lmeval` function can evaluate the model across a broad range of tasks. Here is a comprehensive list of tasks that can be specified:

- 'winogrande'
- 'wsc273'
- 'race'
- 'anli_r1'
- 'anli_r2'
- 'anli_r3'
- 'pubmedqa'
- 'blimp_adjunct_island'
- 'blimp_anaphor_gender_agreement'
- 'blimp_anaphor_number_agreement'
- 'blimp_animate_subject_passive'
- 'blimp_animate_subject_trans'
- 'blimp_causative'
- 'blimp_complex_NP_island'
- 'blimp_coordinate_structure_constraint_complex_left_branch'
- 'blimp_coordinate_structure_constraint_object_extraction'
- 'blimp_determiner_noun_agreement_1'
- 'blimp_determiner_noun_agreement_2'
- 'blimp_determiner_noun_agreement_irregular_1'
- 'blimp_determiner_noun_agreement_irregular_2'
- 'blimp_determiner_noun_agreement_with_adj_2'
- 'blimp_determiner_noun_agreement_with_adj_irregular_1'
- 'blimp_determiner_noun_agreement_with_adj_irregular_2'
- 'blimp_determiner_noun_agreement_with_adjective_1'
- 'blimp_distractor_agreement_relational_noun'
- 'blimp_distractor_agreement_relative_clause'
- 'blimp_drop_argument'
- 'blimp_ellipsis_n_bar_1'
- 'blimp_ellipsis_n_bar_2'
- 'blimp_existential_there_object_raising'
- 'blimp_existential_there_quantifiers_1'
- 'blimp_existential_there_quantifiers_2'
- 'blimp_existential_there_subject_raising'
- 'blimp_expletive_it_object_raising'
- 'blimp_inchoative'
- 'blimp_intransitive'
- 'blimp_irregular_past_participle_adjectives'
- 'blimp_irregular_past_participle_verbs'
- 'blimp_irregular_plural_subject_verb_agreement_1'
- 'blimp_irregular_plural_subject_verb_agreement_2'
- 'blimp_left_branch_island_echo_question'
- 'blimp_left_branch_island_simple_question'
- 'blimp_matrix_question_npi_licensor_present'
- 'blimp_npi_present_1'
- 'blimp_npi_present_2'
- 'blimp_only_npi_licensor_present'
- 'blimp_only_npi_scope'
- 'blimp_passive_1'
- 'blimp_passive_2'
- 'blimp_principle_A_c_command'
- 'blimp_principle_A_case_1'
- 'blimp_principle_A_case_2'
- 'blimp_principle_A_domain_1'
- 'blimp_principle_A_domain_2'
- 'blimp_principle_A_domain_3'
- 'blimp_principle_A_reconstruction'
- 'blimp_regular_plural_subject_verb_agreement_1'
- 'blimp_regular_plural_subject_verb_agreement_2'
- 'blimp_sentential_negation_npi_licensor_present'
- 'blimp_sentential_negation_npi_scope'
- 'blimp_sentential_subject_island'
- 'blimp_superlative_quantifiers_1'
- 'blimp_superlative_quantifiers_2'
- 'blimp_tough_vs_raising_1'
- 'blimp_tough_vs_raising_2'
- 'blimp_transitive'
- 'blimp_wh_island'
- 'blimp_wh_questions_object_gap'
- 'blimp_wh_questions_subject_gap'
- 'blimp_wh_questions_subject_gap_long_distance'
- 'blimp_wh_vs_that_no_gap'
- 'blimp_wh_vs_that_no_gap_long_distance'
- 'blimp_wh_vs_that_with_gap'
- 'blimp_wh_vs_that_with_gap_long_distance'
- 'openbookqa'
- 'arc_easy'
- 'arc_challenge'
- 'sciq'
- 'swag'
- 'piqa'
- 'hellaswag'
- 'glue_mnli'
- 'glue_mnli_mismatched'

This extensive list includes various language understanding and processing tasks, covering different domains and complexities to assess the model comprehensively.
