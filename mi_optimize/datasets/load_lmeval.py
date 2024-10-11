import sys
import random

sys.path.append('./..')
sys.path.append('../..')
sys.path.append('../../..')
import benchmark.lmeval.tasks
import collections
import itertools


def get_testdata(tasks_list, few_shot):

    # replace dataset name
    if 'anli' in tasks_list:
        tasks_list.remove('anli')
        tasks_list += get_anli_tasks_list()
    if 'blimp' in tasks_list:
        tasks_list.remove('blimp')
        tasks_list += get_blimp_tasks_list()
    if 'arc' in tasks_list:
        tasks_list.remove('arc')
        tasks_list += get_arc_tasks_list()
    if 'test_tasks' in tasks_list and len(tasks_list) == 1:
        tasks_list.remove('test_tasks')
        tasks_list += get_test_tasks_list()
    if 'all_tasks_0s' in tasks_list and len(tasks_list) == 1:
        tasks_list.remove('all_tasks_0s')
        tasks_list += get_all_tasks_0s_list()
    if 'all_tasks_5s' in tasks_list and len(tasks_list) == 1:
        tasks_list.remove('all_tasks_5s')
        tasks_list += get_all_tasks_5s_list()
    

    task_dict = benchmark.lmeval.tasks.get_task_dict(tasks_list)
    task_dict_items = [(name, task) for name, task in task_dict.items() if (task.has_validation_docs() or task.has_test_docs())]

    versions = collections.defaultdict(dict)
    requests = collections.defaultdict(list)
    requests_origin = collections.defaultdict(list)

    docs = {}
    for task_name, task in task_dict_items:
        versions[task_name] = task.VERSION
        if task.has_test_docs():
            task_doc_func = task.test_docs
        elif task.has_validation_docs():
            task_doc_func = task.validation_docs
        else:
            raise RuntimeError("Task has neither test_docs nor validation_docs")

        task_docs = list(task_doc_func())
        rnd = random.Random()
        rnd.seed(42)
        rnd.shuffle(task_docs)

        for doc_id, doc in enumerate(itertools.islice(task_docs, 0, None)):
            docs[(task_name, doc_id)] = doc
            ctx = task.fewshot_context(doc=doc, num_fewshot=(5 if few_shot else 0), rnd=rnd, description="")
            reqs = task.construct_requests(doc, ctx)
            if not isinstance(reqs, (list, tuple)):
                reqs = [reqs]
            for i, req in enumerate(reqs):
                requests[req.request_type].append(req)
                requests_origin[req.request_type].append((i, task_name, doc, doc_id))

    return_dict = {"versions": versions, "requests": requests, "requests_origin": requests_origin, "docs": docs}
    return return_dict, task_dict



def get_anli_tasks_list():
    return ['anli_r1', 'anli_r2', 'anli_r3']

def get_blimp_tasks_list():
    blimp_names = ['adjunct_island', 'anaphor_gender_agreement', 'anaphor_number_agreement', 'animate_subject_passive', 'animate_subject_trans', 'causative', 'complex_NP_island', 'coordinate_structure_constraint_complex_left_branch', 'coordinate_structure_constraint_object_extraction', 'determiner_noun_agreement_1', 'determiner_noun_agreement_2', 'determiner_noun_agreement_irregular_1', 'determiner_noun_agreement_irregular_2', 'determiner_noun_agreement_with_adj_2', 'determiner_noun_agreement_with_adj_irregular_1', 'determiner_noun_agreement_with_adj_irregular_2', 'determiner_noun_agreement_with_adjective_1', 'distractor_agreement_relational_noun', 'distractor_agreement_relative_clause', 'drop_argument', 'ellipsis_n_bar_1', 'ellipsis_n_bar_2', 'existential_there_object_raising', 'existential_there_quantifiers_1', 'existential_there_quantifiers_2', 'existential_there_subject_raising', 'expletive_it_object_raising', 'inchoative', 'intransitive', 'irregular_past_participle_adjectives', 'irregular_past_participle_verbs', 'irregular_plural_subject_verb_agreement_1', 'irregular_plural_subject_verb_agreement_2', 'left_branch_island_echo_question', 'left_branch_island_simple_question', 'matrix_question_npi_licensor_present', 'npi_present_1', 'npi_present_2', 'only_npi_licensor_present', 'only_npi_scope', 'passive_1', 'passive_2', 'principle_A_c_command', 'principle_A_case_1', 'principle_A_case_2', 'principle_A_domain_1', 'principle_A_domain_2', 'principle_A_domain_3', 'principle_A_reconstruction', 'regular_plural_subject_verb_agreement_1', 'regular_plural_subject_verb_agreement_2', 'sentential_negation_npi_licensor_present', 'sentential_negation_npi_scope', 'sentential_subject_island', 'superlative_quantifiers_1', 'superlative_quantifiers_2', 'tough_vs_raising_1', 'tough_vs_raising_2', 'transitive', 'wh_island', 'wh_questions_object_gap', 'wh_questions_subject_gap', 'wh_questions_subject_gap_long_distance', 'wh_vs_that_no_gap', 'wh_vs_that_no_gap_long_distance', 'wh_vs_that_with_gap', 'wh_vs_that_with_gap_long_distance']
    blimp_tasks_name = ['blimp_' + x for x in blimp_names]
    return blimp_tasks_name

def get_arc_tasks_list():
    return ['arc_easy', 'arc_challenge']

def get_all_tasks_0s_list():
    task_list = ['winogrande', 'wsc273', 'race', 'anli_r1', 'anli_r2', 'anli_r3', 'pubmedqa', 'openbookqa', 'arc_easy', 'arc_challenge', 'sciq', 'swag', 'piqa', 'hellaswag', 'crows_pairs_english', 'crows_pairs_english_race_color', 'crows_pairs_english_socioeconomic', 'crows_pairs_english_gender', 'crows_pairs_english_age', 'crows_pairs_english_religion', 'crows_pairs_english_disability', 'crows_pairs_english_sexual_orientation', 'crows_pairs_english_nationality', 'crows_pairs_english_physical_appearance', 'crows_pairs_english_autre', 'crows_pairs_french', 'crows_pairs_french_race_color', 'crows_pairs_french_socioeconomic', 'crows_pairs_french_gender', 'crows_pairs_french_age', 'crows_pairs_french_religion', 'crows_pairs_french_disability', 'crows_pairs_french_sexual_orientation', 'crows_pairs_french_nationality', 'crows_pairs_french_physical_appearance', 'crows_pairs_french_autre', 'glue_mnli', 'glue_mnli_mismatched', 'glue_rte', 'glue_qnli', 'glue_sst', 'glue_wnli', 'mathqa', 'mc_taco', 'mutual', 'mutual_plus', 'qa4mre_2011', 'qa4mre_2012', 'qa4mre_2013', 'toxigen']
    return task_list

def get_all_tasks_5s_list():
    task_list = ['winogrande', 'wsc273', 'race', 'anli_r1', 'anli_r2', 'anli_r3', 'pubmedqa', 'blimp_adjunct_island', 'blimp_anaphor_gender_agreement', 'blimp_anaphor_number_agreement', 'blimp_animate_subject_passive', 'blimp_animate_subject_trans', 'blimp_causative', 'blimp_complex_NP_island', 'blimp_coordinate_structure_constraint_complex_left_branch', 'blimp_coordinate_structure_constraint_object_extraction', 'blimp_determiner_noun_agreement_1', 'blimp_determiner_noun_agreement_2', 'blimp_determiner_noun_agreement_irregular_1', 'blimp_determiner_noun_agreement_irregular_2', 'blimp_determiner_noun_agreement_with_adj_2', 'blimp_determiner_noun_agreement_with_adj_irregular_1', 'blimp_determiner_noun_agreement_with_adj_irregular_2', 'blimp_determiner_noun_agreement_with_adjective_1', 'blimp_distractor_agreement_relational_noun', 'blimp_distractor_agreement_relative_clause', 'blimp_drop_argument', 'blimp_ellipsis_n_bar_1', 'blimp_ellipsis_n_bar_2', 'blimp_existential_there_object_raising', 'blimp_existential_there_quantifiers_1', 'blimp_existential_there_quantifiers_2', 'blimp_existential_there_subject_raising', 'blimp_expletive_it_object_raising', 'blimp_inchoative', 'blimp_intransitive', 'blimp_irregular_past_participle_adjectives', 'blimp_irregular_past_participle_verbs', 'blimp_irregular_plural_subject_verb_agreement_1', 'blimp_irregular_plural_subject_verb_agreement_2', 'blimp_left_branch_island_echo_question', 'blimp_left_branch_island_simple_question', 'blimp_matrix_question_npi_licensor_present', 'blimp_npi_present_1', 'blimp_npi_present_2', 'blimp_only_npi_licensor_present', 'blimp_only_npi_scope', 'blimp_passive_1', 'blimp_passive_2', 'blimp_principle_A_c_command', 'blimp_principle_A_case_1', 'blimp_principle_A_case_2', 'blimp_principle_A_domain_1', 'blimp_principle_A_domain_2', 'blimp_principle_A_domain_3', 'blimp_principle_A_reconstruction', 'blimp_regular_plural_subject_verb_agreement_1', 'blimp_regular_plural_subject_verb_agreement_2', 'blimp_sentential_negation_npi_licensor_present', 'blimp_sentential_negation_npi_scope', 'blimp_sentential_subject_island', 'blimp_superlative_quantifiers_1', 'blimp_superlative_quantifiers_2', 'blimp_tough_vs_raising_1', 'blimp_tough_vs_raising_2', 'blimp_transitive', 'blimp_wh_island', 'blimp_wh_questions_object_gap', 'blimp_wh_questions_subject_gap', 'blimp_wh_questions_subject_gap_long_distance', 'blimp_wh_vs_that_no_gap', 'blimp_wh_vs_that_no_gap_long_distance', 'blimp_wh_vs_that_with_gap', 'blimp_wh_vs_that_with_gap_long_distance', 'openbookqa', 'arc_easy', 'arc_challenge', 'sciq', 'swag', 'piqa', 'hellaswag', 'glue_mnli', 'glue_mnli_mismatched', 'glue_rte', 'glue_qnli', 'glue_sst', 'glue_wnli', 'mathqa', 'mc_taco', 'mutual', 'mutual_plus', 'qa4mre_2011', 'qa4mre_2012', 'qa4mre_2013', 'toxigen', 'triviaqa']
    return task_list

def get_test_tasks_list():
    # add your tasks to test
    return []