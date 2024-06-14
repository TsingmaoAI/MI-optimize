import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import datasets

# add winogrande
testdata = datasets.load_dataset(path="winogrande", name="winogrande_xl")
print(testdata)

# add wsc273
testdata = datasets.load_dataset(path="winograd_wsc", name="wsc273")
print(testdata)

# add webqs
testdata = datasets.load_dataset(path="web_questions", name=None)
print(testdata)

# add race
testdata = datasets.load_dataset(path="race", name="high")
print(testdata)

# add anli
testdata = datasets.load_dataset(path="anli", name=None)
print(testdata)

# add squad
testdata = datasets.load_dataset(path="squad_v2", name=None)
print(testdata)

# add lamada
testdata = datasets.load_dataset(path="lambada", name=None)
print(testdata)

# add pubmed_qa
testdata = datasets.load_dataset(path="pubmed_qa", name="pqa_labeled")
print(testdata)

# add blimp
name_list = ['adjunct_island', 'anaphor_gender_agreement', 'anaphor_number_agreement', 'animate_subject_passive', 'animate_subject_trans', 'causative', 'complex_NP_island', 'coordinate_structure_constraint_complex_left_branch', 'coordinate_structure_constraint_object_extraction', 'determiner_noun_agreement_1', 'determiner_noun_agreement_2', 'determiner_noun_agreement_irregular_1', 'determiner_noun_agreement_irregular_2', 'determiner_noun_agreement_with_adj_2', 'determiner_noun_agreement_with_adj_irregular_1', 'determiner_noun_agreement_with_adj_irregular_2', 'determiner_noun_agreement_with_adjective_1', 'distractor_agreement_relational_noun', 'distractor_agreement_relative_clause', 'drop_argument', 'ellipsis_n_bar_1', 'ellipsis_n_bar_2', 'existential_there_object_raising', 'existential_there_quantifiers_1', 'existential_there_quantifiers_2', 'existential_there_subject_raising', 'expletive_it_object_raising', 'inchoative', 'intransitive', 'irregular_past_participle_adjectives', 'irregular_past_participle_verbs', 'irregular_plural_subject_verb_agreement_1', 'irregular_plural_subject_verb_agreement_2', 'left_branch_island_echo_question', 'left_branch_island_simple_question', 'matrix_question_npi_licensor_present', 'npi_present_1', 'npi_present_2', 'only_npi_licensor_present', 'only_npi_scope', 'passive_1', 'passive_2', 'principle_A_c_command', 'principle_A_case_1', 'principle_A_case_2', 'principle_A_domain_1', 'principle_A_domain_2', 'principle_A_domain_3', 'principle_A_reconstruction', 'regular_plural_subject_verb_agreement_1', 'regular_plural_subject_verb_agreement_2', 'sentential_negation_npi_licensor_present', 'sentential_negation_npi_scope', 'sentential_subject_island', 'superlative_quantifiers_1', 'superlative_quantifiers_2', 'tough_vs_raising_1', 'tough_vs_raising_2', 'transitive', 'wh_island', 'wh_questions_object_gap', 'wh_questions_subject_gap', 'wh_questions_subject_gap_long_distance', 'wh_vs_that_no_gap', 'wh_vs_that_no_gap_long_distance', 'wh_vs_that_with_gap', 'wh_vs_that_with_gap_long_distance']
for name in name_list:
    testdata = datasets.load_dataset(path="blimp", name=name)
    print(testdata)

# add gsm8k
testdata = datasets.load_dataset(path="gsm8k", name='main')
print(testdata)

# add openbookqa
testdata = datasets.load_dataset(path="openbookqa", name="main")
print(testdata)

# add ai2_arc_easy
testdata = datasets.load_dataset(path="ai2_arc", name="ARC-Easy")
print(testdata)

# add ai2_arc_challenge
testdata = datasets.load_dataset(path="ai2_arc", name="ARC-Challenge")
print(testdata)

# add sciq
testdata = datasets.load_dataset(path="sciq", name=None)
print(testdata)

# add swag
testdata = datasets.load_dataset(path="swag", name="regular")
print(testdata)

# add wikitext
testdata = datasets.load_dataset(path="EleutherAI/wikitext_document_level", name="wikitext-2-raw-v1")
print(testdata)

# add piqa
try:
    testdata = datasets.load_dataset(path="piqa", name=None, encoding='ISO-8859-1')
except Exception as e:
    print("Load Error: piqa")
    testdata = datasets.load_dataset(path="piqa", name=None)
    print(testdata)

# add hellaswag
try:
    testdata = datasets.load_dataset(path="hellaswag", name=None, encoding='ISO-8859-1')
except Exception as e:
    print("Load Error: hellaswag")
    testdata = datasets.load_dataset(path="hellaswag", name=None)
    print(testdata)
    
name_list = ['arithmetic_2da', 'arithmetic_2ds', 'arithmetic_3da', 'arithmetic_3ds', 'arithmetic_4da', 'arithmetic_4ds', 'arithmetic_5da', 'arithmetic_5ds', 'arithmetic_2dm', 'arithmetic_1dc']
for name in name_list:
    try:
        testdata = datasets.load_dataset(path="EleutherAI/arithmetic", name=name, encoding='ISO-8859-1')
    except Exception as e:
        print("Load Error: EleutherAI/arithmetic + ", name)
        testdata = datasets.load_dataset(path="EleutherAI/arithmetic", name=name)
        print(testdata)

try:
    testdata = datasets.load_dataset(path="hendrycks_test", name=None, encoding='ISO-8859-1')
except Exception as e:
    print("Load Error: hendrycks_test")
name_list = ['anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']
for name in name_list:
    testdata = datasets.load_dataset(path="hendrycks_test", name=name)
    print(testdata)


try:
    testdata = datasets.load_dataset(path="BigScienceBiasEval/crows_pairs_multilingual", name=None, encoding='ISO-8859-1')
except Exception as e:
    print("Load Error: BigScienceBiasEval/crows_pairs_multilingual + ", None)
try:
    testdata = datasets.load_dataset(path="BigScienceBiasEval/crows_pairs_multilingual", name=None)
    print(testdata)
except Exception as e:
    print("Decode Error: BigScienceBiasEval/crows_pairs_multilingual + ", None)

# add glue
name_list = ['cola', 'sst2', 'mnli', 'qnli', 'wnli', 'rte', 'mrpc', 'qqp', 'stsb']
for name in name_list:
    testdata = datasets.load_dataset(path="glue", name=name)
    print(testdata)

# TODO(pshenhao): add lambada_openai
try:
    testdata = datasets.load_dataset(path="EleutherAI/lambada_openai", name=None, encoding='ISO-8859-1')
except Exception as e:
    print("Load Error: EleutherAI/lambada_openai + ", None)
try:
    testdata = datasets.load_dataset(path="EleutherAI/lambada_openai", name=None)
    print(testdata)
except Exception as e:
    print("Decode Error: EleutherAI/lambada_openai + ", None)

# add math_qa
try:
    testdata = datasets.load_dataset(path="math_qa", name=None, encoding='ISO-8859-1')
except Exception as e:
    print("Load Error: math_qa + ", None)
try:
    testdata = datasets.load_dataset(path="math_qa", name=None)
    print(testdata)
except Exception as e:
    print("Decode Error: math_qa + ", None)

# add mc_taco
try:
    testdata = datasets.load_dataset(path="mc_taco", name=None, encoding='ISO-8859-1')
except Exception as e:
    print("Load Error: mc_taco + ", None)
try:
    testdata = datasets.load_dataset(path="mc_taco", name=None)
    print(testdata)
except Exception as e:
    print("Decode Error: mc_taco + ", None)

# add qa4mre
try:
    testdata = datasets.load_dataset(path="qa4mre", name=None, encoding='ISO-8859-1')
except Exception as e:
    print("Load Error: qa4mre + ", None)
name_list = ['2011.main.DE', '2011.main.EN', '2011.main.ES', '2011.main.IT', '2011.main.RO', '2012.main.AR', '2012.main.BG', '2012.main.DE', '2012.main.EN', '2012.main.ES', '2012.main.IT', '2012.main.RO', '2012.alzheimers.EN', '2013.main.AR', '2013.main.BG', '2013.main.EN', '2013.main.ES', '2013.main.RO', '2013.alzheimers.EN', '2013.entrance_exam.EN']
for name in name_list:
    testdata = datasets.load_dataset(path="qa4mre", name=name)
    print(testdata)

# add qasper
try:
    testdata = datasets.load_dataset(path="qasper", name=None, encoding='ISO-8859-1')
except Exception as e:
    print("Load Error: qasper + ", None)
testdata = datasets.load_dataset(path="qasper", name=None)
print(testdata)

try:
    testdata = datasets.load_dataset(path="story_cloze", name=None, encoding='ISO-8859-1')
except Exception as e:
    print("Load Error: story_cloze + ", None)
testdata = datasets.load_dataset("story_cloze", '2016')
testdata = datasets.load_dataset("story_cloze", '2018')
print(testdata)

# add toxigen
testdata = datasets.load_dataset(path="skg/toxigen-data", name="annotated")
print(testdata)

