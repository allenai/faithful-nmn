import sys
import json
import numpy as np
from tqdm import tqdm


def calc_f1(precision, recall):
    return 2.0 * precision * recall / (precision + recall)


def method1_calc_statistic(scores):
    precisions = [
        precision_good / float(precision_good + precision_bad)
        for [precision_good, precision_bad, _, _] in scores
    ]
    recalls = [
        recall_good / float(recall_good + recall_bad)
        for [_, _, recall_good, recall_bad] in scores
    ]
    f1s = [calc_f1(precision, recall) for precision, recall in zip(precisions, recalls)]
    score1, score2 = f1s
    statistic = f1s[0] - f1s[1]
    return score1, score2, statistic


# Input format: python significance_test.py predictions1.json predictions2.json sample_size
# This tests the null hypothesis that module_accuracy(predictions1) == module_accuracy(predictions2) against
# the alternative hypothesis that module_accuracy(predictions1) > module_accuracy(predictions2)
# sample_size is an integer specifying the number of permutations that will be sampled to conduct this test

f1 = open(sys.argv[1])
f2 = open(sys.argv[2])
sample_size = int(sys.argv[3])
avg_method = int(sys.argv[4])
lines1 = f1.readlines()
lines2 = f2.readlines()
predictions1 = [json.loads(line) for line in lines1]
predictions2 = [json.loads(line) for line in lines2]
assert len(predictions1) == len(predictions2)
predictions = [predictions1, predictions2]

if avg_method == 1:
    # For each model, we store all of the precision good/bad counts and the recall good/bad counts
    scores = [[0, 0, 0, 0] for _ in range(2)]
else:
    scores = [[], []]
# i indexes which model (there are 2 models we are comparing)
for i in range(2):
    # Iterating through each instance
    for j, pred in enumerate(predictions[i]):
        if avg_method == 1:
            precision_good, precision_bad, recall_good, recall_bad = pred[
                "box_f1_overall_score"
            ]
            scores[i][0] += precision_good
            scores[i][1] += precision_bad
            scores[i][2] += recall_good
            scores[i][3] += recall_bad
        elif avg_method == 2:
            scores[i].append(pred["box_f1_overall_score"])
        elif avg_method == 3:
            # Iterating through each module for which there is at least 1 gold box
            # pair = (a,b), where a is the index of a module in the program, and
            # b indexes image 0 or image 1
            # Previously used for mAP:
            for pair in pred["box_f1"]["non_empty_modules"]:
                scores[i].append(pred["box_f1"][str(pair[0])][pair[1]])
        assert predictions[i][j]["image_id"] == predictions[1 - i][j]["image_id"]
        assert predictions[i][j]["utterance"] == predictions[1 - i][j]["utterance"]
if avg_method == 1:
    score1, score2, original_statistic = method1_calc_statistic(scores)
    print("Score 1: " + str(score1))
    print("Score 2: " + str(score2))
elif avg_method in {2, 3}:
    assert len(scores[0]) == len(scores[1])
    original_statistic = np.mean(scores[0]) - np.mean(scores[1])
    print("Score 1: " + str(np.mean(scores[0])))
    print("Score 2: " + str(np.mean(scores[1])))
print("Original statistic: " + str(original_statistic))

num_exceeding = 0
# Sampling permutations, a permutation consists of a 0/1 for each module in the scores
# list
num_examples = len(scores[0])
if avg_method == 1:
    num_examples = len(predictions[0])
signs = np.random.binomial(1, 0.5, size=(sample_size, num_examples))
for trial in tqdm(range(signs.shape[0])):
    if avg_method == 1:
        scores = [[0, 0, 0, 0] for _ in range(2)]
    else:
        scores = [[], []]
    for i in range(2):
        # count indexes which of the 0/1 values in the permutation should be used for a given
        # module
        count = 0
        # Previously used for mAP:
        for j, pred in enumerate(predictions[i]):
            if avg_method == 1:
                if signs[trial][count] >= 0.5:
                    for k in range(len(pred["box_f1_overall_score"])):
                        scores[i][k] += pred["box_f1_overall_score"][k]
                else:
                    for k in range(len(pred["box_f1_overall_score"])):
                        scores[1 - i][k] += pred["box_f1_overall_score"][k]
                count += 1
            elif avg_method == 2:
                if signs[trial][count] >= 0.5:
                    scores[i].append(pred["box_f1_overall_score"])
                else:
                    scores[1 - i].append(pred["box_f1_overall_score"])
                count += 1
            elif avg_method == 3:
                for pair in pred["box_f1"]["non_empty_modules"]:
                    if signs[trial][count] >= 0.5:
                        scores[i].append(pred["box_f1"][str(pair[0])][pair[1]])
                    else:
                        scores[1 - i].append(pred["box_f1"][str(pair[0])][pair[1]])
                    count += 1
    if avg_method == 1:
        _, _, statistic = method1_calc_statistic(scores)
    elif avg_method in {2, 3}:
        assert len(scores[0]) == len(scores[1])
        statistic = np.mean(scores[0]) - np.mean(scores[1])
    if abs(statistic) >= original_statistic:
        num_exceeding += 1

print("p-value: " + str((1.0 + float(num_exceeding)) / (1.0 + signs.shape[0])))
