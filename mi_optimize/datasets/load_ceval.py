import random
import re
from itertools import islice
import os

from matplotlib import category
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from datasets import load_dataset

def get_ceval_mapping():
    TASK2CTG = {
        'computer_network': 'STEM',
        'operating_system': 'STEM',
        'computer_architecture': 'STEM',
        'college_programming': 'STEM',
        'college_physics': 'STEM',
        'college_chemistry': 'STEM',
        'advanced_mathematics': 'STEM',
        'probability_and_statistics': 'STEM',
        'discrete_mathematics': 'STEM',
        'electrical_engineer': 'STEM',
        'metrology_engineer': 'STEM',
        'high_school_mathematics': 'STEM',
        'high_school_physics': 'STEM',
        'high_school_chemistry': 'STEM',
        'high_school_biology': 'STEM',
        'middle_school_mathematics': 'STEM',
        'middle_school_biology': 'STEM',
        'middle_school_physics': 'STEM',
        'middle_school_chemistry': 'STEM',
        'veterinary_medicine': 'STEM',
        'college_economics': 'Social Science',
        'business_administration': 'Social Science',
        'marxism': 'Social Science',
        'mao_zedong_thought': 'Social Science',
        'education_science': 'Social Science',
        'teacher_qualification': 'Social Science',
        'high_school_politics': 'Social Science',
        'high_school_geography': 'Social Science',
        'middle_school_politics': 'Social Science',
        'middle_school_geography': 'Social Science',
        'modern_chinese_history': 'Humanities',
        'ideological_and_moral_cultivation': 'Humanities',
        'logic': 'Humanities',
        'law': 'Humanities',
        'chinese_language_and_literature': 'Humanities',
        'art_studies': 'Humanities',
        'professional_tour_guide': 'Humanities',
        'legal_professional': 'Humanities',
        'high_school_chinese': 'Humanities',
        'high_school_history': 'Humanities',
        'middle_school_history': 'Humanities',
        'civil_servant': 'Other',
        'sports_science': 'Other',
        'plant_protection': 'Other',
        'basic_medicine': 'Other',
        'clinical_medicine': 'Other',
        'urban_and_rural_planner': 'Other',
        'accountant': 'Other',
        'fire_engineer': 'Other',
        'environmental_impact_assessment_engineer': 'Other',
        'tax_accountant': 'Other',
        'physician': 'Other',
    }

    return TASK2CTG


def get_subjects_ceval(subject_name):
    TASK2CTG = get_ceval_mapping()
    if subject_name == 'hm':
        subjects_dict = {key: value for key, value in TASK2CTG.items() if value == 'Humanities'}
    elif subject_name == 'st':
        subjects_dict = {key: value for key, value in TASK2CTG.items() if value == 'STEM'}
    elif subject_name == 'ss':
        subjects_dict = {key: value for key, value in TASK2CTG.items() if value == 'Social Science'}
    else:
        subjects_dict = TASK2CTG
    return subjects_dict


def load_ceval(subjects, split, path="ceval/ceval-exam"):
    # print("get_ceval")
    datasets = {}
    for sub in subjects:
        datasets[sub] = load_dataset(path=path, name=sub)
        datasets[sub] = datasets[sub][split]

    return datasets


def get_ceval(subject='all', split='test', question=4, shuffle=False, seed=42, answer=False):
    """
        @qustion: Number of questions per discipline
        @shuffle: Is it randomly shuffled (Whether to randomly select questions for each discipline)
        @seed: random seed
        @answer: if include the answer
    """

    TASK2CTG = get_ceval_mapping()
    if subject == 'all':
        subjects = [key for key, _ in TASK2CTG.items()]
    elif subject in ['STEM', 'Social Science', 'Humanities', 'Other']:
        subjects = [key for key, value in TASK2CTG.items() if value == subject]
    else:
        subjects = subject
    ceval_data = load_ceval(subjects=subjects, split=split)
    ceval_question_list = []
    for subject_name in ceval_data:
        subject_data = ceval_data[subject_name]
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
            question_str = data['question']
            choices_str = '\n'.join([f"{choice}. {data[choice]}" for choice in ['A', 'B', 'C', 'D']])
            answer_str = f"{data['answer']}" if answer else ""
            ceval_question_str = (f"{question_str}\n{choices_str}\n答案：{answer_str}")
            ceval_question_list.append(ceval_question_str)
            
    return ceval_question_list


def get_calibrate_ceval(tokenizer, calibrate_subject='all', split='test', calibrate_nums=4, shuffle=False, seed=42, answer=False, calibrate_seqlen=2048, **kwargs):
    calibrate_data = get_ceval(subject=calibrate_subject, split=split, question=calibrate_nums, shuffle=shuffle, seed=seed, answer=answer)
    inputs_ids = []
    for data in calibrate_data:
        input_ids = tokenizer.encode(data, return_tensors='pt')
        inputs_ids.append(input_ids[:calibrate_seqlen])
    return inputs_ids


def get_fewshot_ceval(subject='all', split='test', question=5, shuffle=False, seed=42, answer=True, model_name=""):
    ceval_content_list = get_ceval(subject=subject, split=split, question=question, shuffle=shuffle, seed=seed, answer=answer)

    title = f"以下是中国考试的单项选择题，请选出其中的正确答案。"
    if model_name == "chatglm":
        prompt = []
        for content in ceval_content_list:
            prompt.append({'role': 'user', 'content': f"{title}\n{content[:-1]}"})
            prompt.append({'role': 'assistant', 'metadata': '', 'content': content[-1]})

    elif model_name == "baichuan" or model_name == 'llama':
        prompt = title + "\n"
        for content in ceval_content_list:
            prompt = prompt + "\n\n" + content

    else:
        prompt = None

    return prompt

def get_testdaset_ceval(subject='all', split='test', question='all', shuffle=False, seed=42, answer=True):
    ceval_question_list = get_ceval(subject=subject, split=split, question=question, shuffle=shuffle, seed=seed, answer=answer)
    
    question_list = []
    answer_list = []
    for ceval_question in ceval_question_list:
        question_list.append(ceval_question[:-1])
        answer_list.append(ceval_question[-1].strip().upper())
    
    return question_list, answer_list


def extract_cot_answer_ceval(question, response):
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


def classifi_results_ceval(results):
    TASK2CTG = get_ceval_mapping()
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