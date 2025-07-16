import csv
import math
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import json


def store_json(d, path):
    with open(path, 'a+', encoding='utf-8') as f:
        json.dump(d, f)


def add_to_json(d, path):
    with open(path, 'r+', encoding='utf-8') as f:
        curr_data = json.load(f)
    if isinstance(curr_data, list):
        new_data = curr_data + d
    elif isinstance(curr_data, dict):
        curr_data.update(d)
    with open(path, 'w+', encoding='utf-8') as f:
        json.dump(new_data, f)


def process_generation(text: str):  # diffrence between this and normlize text?? ask roi
    if not text:
        return text
    while text and text[0] in ['\n', ':', ' ', ',', ';']:
        text = text[1:]
    # if '\n' in text and text.index('\n') >= 4:
    #     text = text[:text.index('\n')]
    # text.replace('\n', ' ')
    return text


def write_to_csv(path: str, table: list):
    with open(path, 'a+', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        for line in table:
            csv_writer.writerow(line)


def read_from_csv(path: str):
    table = []
    with open(path, 'r+', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        for line in csv_reader:
            table.append(line)
    return table


def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))


def check_answer_truthfulness(generated_answer, gold_answers):
    if isinstance(gold_answers, str):
        gold_answers = [gold_answers]
    normalized_generation = normalize_text(generated_answer)
    return any([normalize_text(answer) in normalized_generation for answer in gold_answers])


def is_answer_dont_know(answer):
    normalized_text = normalize_text(answer)
    return any([normalize_text(option) in normalized_text for option in
                ["I don't know", "I do not know", "i don't know", "i do not know"]])


def get_AUROC(cached_results, index_truth, index_decision, is_binary=True):
    true_labels = []
    predicted_scores = []
    for row in cached_results:
        if len(row) == 0:
            continue
        true_label = row[index_truth]
        decision_label = row[index_decision]
        true_labels.append(true_label)
        predicted_scores.append(decision_label)

    true_labels = [True if (s == 'True' or s == 'TRUE') else False for s in true_labels]
    if (is_binary):
        predicted_scores = [True if (s == 'True' or s == 'TRUE') else False for s in predicted_scores]
        predicted_scores = np.array(predicted_scores).astype(int)
    else:
        predicted_scores = np.array(predicted_scores).astype(np.double)

    true_labels = np.array(true_labels).astype(int)
    if (np.all(true_labels) or not np.any(true_labels)):
        return "cant calc AURROC" if is_binary else "cant", "calc", "AURROC"
    auroc = roc_auc_score(true_labels, predicted_scores)
    if (not is_binary):
        return auroc, true_labels, predicted_scores

    return auroc


def brier_score(cached_results, index_truth, index_decision, is_binary=True):
    true_labels = []
    predicted_scores = []
    for row in cached_results:
        if len(row) == 0:
            continue
        true_label = row[index_truth]
        decision_label = row[index_decision]
        true_labels.append(true_label)
        predicted_scores.append(decision_label)

    true_labels = [True if s == 'True' else False for s in true_labels]
    true_labels = np.array(true_labels).astype(int)

    if (is_binary):
        predicted_scores = [True if s == 'True' else False for s in predicted_scores]
        predicted_scores = np.array(predicted_scores).astype(int)
    else:
        predicted_scores = np.array(predicted_scores).astype(np.double)

    n = len(true_labels)
    score = 0.0
    for i in range(n):
        diff = predicted_scores[i] - true_labels[i]
        score += diff ** 2
    score /= n
    return score
