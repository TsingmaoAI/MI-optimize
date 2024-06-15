import re
import string
from collections import Counter


def normalize_answer(task,s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    
    answer = white_space_fix(remove_articles(remove_punc(lower(s))))

    if task == "NaturalLanguageInference":
        if answer == "ent":
            answer = "entailment"
    elif task == "ToxicDetection":
        if answer == "ben":
            answer = "benign"
        elif answer == "to":
            answer = "toxic"

    return answer


def f1_score(task,prediction, ground_truth):
    prediction_tokens = normalize_answer(task,prediction).split()
    ground_truth_tokens = normalize_answer(task,ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match_score(task,prediction, ground_truth):
    return normalize_answer(task,prediction) == normalize_answer(task,ground_truth)

def metric_max_over_ground_truths(task,metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(task,prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

from collections import defaultdict
def evaluate_ner(predictions, references):
    entity_types = set([item[1] for sublist in references for item in sublist])
    scores = defaultdict(dict)
    for entity_type in entity_types:
        # Convert prediction and ground-truth to sets of tuples for the current entity type
        prediction = {tuple(item) for sublist in predictions for item in sublist if item[1] == entity_type}
        ground_truth = {tuple(item) for sublist in references for item in sublist if item[1] == entity_type}

        # Calculate true positives, false positives, and false negatives for the current entity type
        tp = len(prediction.intersection(ground_truth))
        fp = len(prediction - ground_truth)
        fn = len(ground_truth - prediction)

        # Calculate precision, recall, and F1-score for the current entity type
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        # Store the scores for the current entity type
        scores[entity_type]['precision'] = precision
        scores[entity_type]['recall'] = recall
        scores[entity_type]['f1_score'] = f1_score

    # Calculate the average F1-score across all entity types
    avg_f1_score = sum([scores[entity_type]['f1_score'] for entity_type in entity_types]) / len(entity_types)

    # Return the scores as a dictionary
    scores['avg'] = {'precision': 0, 'recall': 0, 'f1_score': avg_f1_score}
    return scores

def compute_metric(task, predictions, references):
    if task in ["SentimentAnalysis", "ToxicDetection", "NaturalLanguageInference"]:
        # results = 100.00 * sum([1 if normalize_answer(pred) == ref else 0 for pred, ref in zip(predictions, references)]) / len(references)
        sum = 0
        for pred, ref in zip(predictions, references):
            print("pred:"+normalize_answer(task,pred))
            print("ref:"+ref)
            if normalize_answer(task,pred) == ref:
                sum += 1
        results = sum / len(references) *100
    
    elif task == "NameEntityRecognition":
        from datasets import load_metric
        results = 100.00 * evaluate_ner(predictions, references)["avg"]["f1_score"]
    elif task == "QuestionAnswering":
        em, f1 = 0, 0
        for pred, ground_truth in zip(predictions, references):
            em += metric_max_over_ground_truths(task,exact_match_score, pred, ground_truth)
            f1 += metric_max_over_ground_truths(task,f1_score, pred, ground_truth)
        results = {"exact_match": 100.00 * em / len(references), "f1": 100.00 * f1 / len(references)}
    return results
