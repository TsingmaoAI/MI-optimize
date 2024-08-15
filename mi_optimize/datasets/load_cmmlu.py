import re
import csv
import random
from itertools import islice

def get_cmmlu_mapping():
    TASK2CTG = {
        'agronomy': 'Other',
        'anatomy': 'STEM',
        'ancient_chinese': 'China Specific',
        'arts': 'Humanities',
        'astronomy': 'STEM',
        'business_ethics': 'Social Sciences',
        'chinese_civil_service_exam': 'China Specific',
        'chinese_driving_rule': 'China Specific',
        'chinese_food_culture': 'China Specific',
        'chinese_foreign_policy': 'China Specific',
        'chinese_history': 'China Specific',
        'chinese_literature': 'China Specific',
        'chinese_teacher_qualification': 'China Specific',
        'clinical_knowledge': 'Other',
        'college_actuarial_science': 'STEM',
        'college_education': 'Social Sciences',
        'college_engineering_hydrology': 'STEM',
        'college_law': 'Humanities',
        'college_mathematics': 'STEM',
        'college_medical_statistics': 'STEM',
        'college_medicine': 'Other',
        'computer_science': 'STEM',
        'computer_security': 'Other',
        'conceptual_physics': 'STEM',
        'construction_project_management': 'China Specific',
        'economics': 'Social Sciences',
        'education': 'Social Sciences',
        'electrical_engineering': 'STEM',
        'elementary_chinese': 'China Specific',
        'elementary_commonsense': 'China Specific',
        'elementary_information_and_technology': 'Other',
        'elementary_mathematics': 'STEM',
        'ethnology': 'China Specific',
        'food_science': 'Other',
        'genetics': 'STEM',
        'global_facts': 'Humanities',
        'high_school_biology': 'STEM',
        'high_school_chemistry': 'STEM',
        'high_school_geography': 'Social Sciences',
        'high_school_mathematics': 'STEM',
        'high_school_physics': 'STEM',
        'high_school_politics': 'China Specific',
        'human_sexuality': 'Other',
        'international_law': 'Humanities',
        'journalism': 'Social Sciences',
        'jurisprudence': 'Humanities',
        'legal_and_moral_basis': 'Other',
        'logical': 'Humanities',
        'machine_learning': 'STEM',
        'management': 'Social Sciences',
        'marketing': 'Social Sciences',
        'marxist_theory': 'Humanities',
        'modern_chinese': 'China Specific',
        'nutrition': 'Other',
        'philosophy': 'Humanities',
        'professional_accounting': 'Social Sciences',
        'professional_law': 'Humanities',
        'professional_medicine': 'Other',
        'professional_psychology': 'Social Sciences',
        'public_relations': 'Social Sciences',
        'security_study': 'Social Sciences',
        'sociology': 'Social Sciences',
        'sports_science': 'Other',
        'traditional_chinese_medicine': 'China Specific',
        'virology': 'STEM',
        'world_history': 'Humanities',
        'world_religions': 'Humanities'
    }

    return TASK2CTG

def get_subjects_cmmlu(subject_name):
    TASK2CTG = get_cmmlu_mapping()
    if subject_name == 'hm':
        subjects_dict = {key: value for key, value in TASK2CTG.items() if value == 'Humanities'}
    elif subject_name == 'st':
        subjects_dict = {key: value for key, value in TASK2CTG.items() if value == 'STEM'}
    elif subject_name == 'ss':
        subjects_dict = {key: value for key, value in TASK2CTG.items() if value == 'Social Sciences'}
    else:
        subjects_dict = TASK2CTG
    return subjects_dict

def load_cmmlu(subjects, split, path="mi_optimize/datasets/cmmlu_v1_0_1"):
    # print("get CMMLU")

    datasets = {}
    for sub in subjects:
        datasets[sub] = []
        sub_path = path + r"/" + split + r"/" + sub + ".csv"
        with open(sub_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                datasets[sub].append(dict(row))

    return datasets


def get_cmmlu(subject='all', split='test', question=4, shuffle=False, seed=42, answer=False):
    """
        @qustion: Number of questions per discipline
        @shuffle: Is it randomly shuffled (Whether to randomly select questions for each discipline)
        @seed: random seed
        @answer: if include the answer
    """

    TASK2CTG = get_cmmlu_mapping()
    if subject == 'all':
        subjects = [key for key, _ in TASK2CTG.items()]
    elif subject in ['STEM', 'Social Sciences', 'Humanities', 'China Specific', 'Other']:
        subjects = [key for key, value in TASK2CTG.items() if value == subject]
    else:
        subjects = subject

    CMMLU_data = load_cmmlu(subjects=subjects, split=split)
    CMMLU_question_list = []
    for subject_name in CMMLU_data:
        subject_data = CMMLU_data[subject_name]

        # if get all question
        if question == 'all':
            question_num = len(subject_data)
        elif isinstance(question, int):
            question_num = len(subject_data) if question > len(subject_data) else question
        else:
            question_num = len(subject_data) if question[subject_name] > len(subject_data) else question[subject_name]  # Limit quantity

        # if shuffle
        if shuffle:
            random.seed(seed)
            subject_data = list(subject_data)
            random.shuffle(subject_data)
        
        # Construct calibration problem list
        for data in islice(subject_data, question_num):
            question_str = data['Question']
            choices_str = '\n'.join([f"{choice}. {data[choice]}" for choice in ['A', 'B', 'C', 'D']])
            answer_str = f"{data['Answer']}" if answer else ""
            CMMLU_question_str = (f"{question_str}\n{choices_str}\n答案：{answer_str}")
            CMMLU_question_list.append(CMMLU_question_str)
            
    return CMMLU_question_list

def get_calibrate_cmmlu(tokenizer, calibrate_subject='all', split='test', calibrate_nums=4, shuffle=False, seed=42, answer=False, calibrate_seqlen=2048, **kwargs):
    calibrate_data = get_cmmlu(subject=calibrate_subject, split=split, question=calibrate_nums, shuffle=shuffle, seed=seed, answer=answer)
    inputs_ids = []
    for data in calibrate_data:
        input_ids = tokenizer.encode(data, return_tensors='pt')
        inputs_ids.append(input_ids[:calibrate_seqlen])
    return inputs_ids
        

def get_fewshot_cmmlu(subject='all', split='test', num_shot=5, shuffle=False, seed=42, answer=True, model_type=""):
    title = f"以下是中国考试的单项选择题，请选出其中的正确答案。"
    if num_shot == 0:
        if model_type == "chatglm" or model_type == 'qwen':
            prompt = []
        elif model_type == "baichuan" or model_type == 'llama':
            prompt = ""
    
    else:
        cmmlu_content_list = get_cmmlu(subject=[subject], split=split, question=num_shot, shuffle=shuffle, seed=seed, answer=answer)
        if model_type == "chatglm":
            prompt = []
            for content in cmmlu_content_list:
                prompt.append({'role': 'user', 'content': f"{title}\n{content[:-1]}"})
                prompt.append({'role': 'assistant', 'metadata': '', 'content': content[-1]})

        elif model_type == "baichuan" or model_type == "llama":
            prompt = title + "\n"
            for content in cmmlu_content_list:
                prompt = prompt + "\n\n" + content

        elif model_type == "qwen":
            prompt = []
            for content in cmmlu_content_list:
                prompt.append({'role': 'user', 'content': f"{title}\n{content[:-1]}"})
                prompt.append({'role': 'system', 'content': content[-1]})            


    return prompt


def get_testdata_cmmlu(subject='all', split='test', question='all', shuffle=False, seed=42, answer=True):
    CMMLU_question_list = get_cmmlu(subject=subject, split=split, question=question, shuffle=shuffle, seed=seed, answer=answer)
    
    question_list = []
    answer_list = []
    for CMMLU_question in CMMLU_question_list:
        question_list.append(CMMLU_question[:-1])
        answer_list.append(CMMLU_question[-1].strip().upper())
    
    return question_list, answer_list


def extract_cot_answer_cmmlu(question, response):
    choices_list = ['A', 'B', 'C', 'D']
    m = re.findall(r'所以答案是(.+?)。', response, re.M)
    if len(m) > 0 and m[-1] in choices_list:
        return m[-1]
    
    # RE extraction
    answer_patterns = [
        r'([ABCD])是正确的',
        r'选项([ABCD])正确',
        r'答案为([ABCD])',
        r'答案是([ABCD])',
        r'答案([ABCD])',
        r'选择([ABCD])',
        r'答案：([ABCD])',
        r'选择答案([ABCD])'
    ]
    for answer_pattern in answer_patterns:
        m = re.search(answer_pattern, response, re.M)
        if m:
            answer = m.group(1)
            return answer
        
    # only containing one choice-character
    m = re.findall(r'[ABCD]', response, re.M)
    if len(m) == 1:
        answer = m[0]
        return answer
    answer_word_counter = 0

    # only containing one choice-context
    options = re.findall(r'\b[A-D]\.\s*(.*?)\n', question)
    option_dict = {choice:option for choice, option in zip(choices_list, options)}
    for c in choices_list:
        if str(option_dict[f'{c}']) in response:
            answer = c
            answer_word_counter += 1
    if answer_word_counter == 1:
        return answer
    
    return '-'


def classifi_results_cmmlu(results):
    TASK2CTG = get_cmmlu_mapping()
    all_results = {}
    all_results['subjects'] = {}
    all_results['categories'] = {}
    
    for sub, stats in results.items():
        total = int(stats.split(' ')[3])
        correct = int(stats.split(' ')[1])
        ratio = float(stats.split(' ')[5])

        if sub in TASK2CTG:
            category = TASK2CTG[sub]
            if category not in all_results['categories']:
                all_results['categories'][category] = {'total': total, 'correct': correct}
            else:
                all_results['categories'][category]['total'] += total
                all_results['categories'][category]['correct'] += correct
            all_results['subjects'][sub] = {'total': total, 'correct': correct, 'ratio': ratio}

    for category in all_results['categories']:
        all_results['categories'][category]['ratio'] = all_results['categories'][category]['correct'] / all_results['categories'][category]['total']

    return all_results