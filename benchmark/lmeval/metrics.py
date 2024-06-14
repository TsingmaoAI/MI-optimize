
import math
import numpy as np

def yesno(x):
    if x:
        return "yes"
    else:
        return "no"

def mean(arr):
    return sum(arr) / len(arr)

def perplexity(items):
    return math.exp(-mean(items))

def weighted_mean(items):
    a, b = zip(*items)
    return sum(a) / sum(b)

def weighted_perplexity(items):
    return math.exp(-weighted_mean(items))

def bits_per_byte(items):
    return -weighted_mean(items) / math.log(2)

# def matthews_corrcoef(items):
#     unzipped_list = list(zip(*items))
#     golds = unzipped_list[0]
#     preds = unzipped_list[1]
#     return sklearn.metrics.matthews_corrcoef(golds, preds)

# def f1_score(items):
#     unzipped_list = list(zip(*items))
#     golds = unzipped_list[0]
#     preds = unzipped_list[1]
#     fscore = sklearn.metrics.f1_score(golds, preds)

#     return np.max(fscore)