from tqdm import tqdm
from mi_optimize.datasets.load_ceval import get_subjects_ceval, get_testdaset_ceval

results = {}
subject = 'all'
subject_dict = get_subjects_ceval(subject)
question_list, answer_list = get_testdaset_ceval(subject=subject_dict, split='test')
for subject in tqdm(subject_dict):
    count = 0
    correct = 0
    for question, answer in zip(question_list, answer_list):
        pass