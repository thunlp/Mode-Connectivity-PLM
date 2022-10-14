import numpy as np
import string
import re
from collections import Counter
from sklearn.metrics import matthews_corrcoef, f1_score
from scipy.stats import pearsonr
from rouge import Rouge
from collections import OrderedDict


def itp_evaluate(logger, predictions, data, metric):
    def cast_to_float(predictions):
        new_predictions = []
        for prediction in predictions:
            try:
                new_predictions.append(float(prediction.strip()))
            except:
                new_predictions.append(float('NaN'))
        assert len(new_predictions) == len(predictions)
        return new_predictions

    assert len(predictions) == len(data)
    return_dict = OrderedDict()
    if metric == "EM":
        logger.info("======into EM, print answers=========")
        for dp in data[:5]:
            logger.info(dp[1])
        ems = []
        for (prediction, dp) in zip(predictions, data):
            ems.append(get_exact_match_over_list(prediction, dp[1]))
        return_dict[metric] = np.mean(ems)
        # return np.mean(ems)
    elif metric == "ACC":
        accs = []
        logger.info("======into ACC, print answers=========")
        for dp in data[:5]:
            logger.info(dp[1])
        for (prediction, dp) in zip(predictions, data):
            if_correct = get_accruacy_over_list(prediction, dp[1])
            accs.append(if_correct)
        return_dict[metric] = np.mean(accs)
        return_dict["raw_score"] = accs

    elif metric == "QA-F1":
        f1s = []
        logger.info("======into QA-F1, print answers=========")
        for dp in data[:5]:
            logger.info(dp[1])
        for (prediction, dp) in zip(predictions, data):
            f1s.append(get_f1_over_list(prediction, dp[1]))
        return_dict[metric] = np.mean(f1s)
        return_dict["raw_score"] = f1s
        # return np.mean(f1s)
    elif metric == "Classification-F1":
        logger.info("======into Claasification F1, print answers=========")
        for dp in data[:5]:
            logger.info(dp[1][0])
        accs = []
        for (prediction, dp) in zip(predictions, data):
            if_correct = get_accruacy_over_list(prediction, dp[1])
            accs.append(if_correct)
        return_dict[metric] = f1_score(
            [dp[1][0] for dp in data], predictions, average="macro", labels=np.unique([dp[1][0] for dp in data]))
        return_dict["raw_score"] = accs

    elif metric == "Matthew-Correlation":
        return_dict[metric] = get_matthews_corr(data, predictions)
    elif metric == "Pearson-Correlation":
        predictions = cast_to_float(predictions)
        return_dict[metric] = pearsonr(
            [float(dp[1][0]) for dp in data], predictions)[0]
    elif metric == "Rouge-L":
        rouges = []
        for (prediction, dp) in zip(predictions, data):
            rouges.append(get_rouge_over_list(prediction, dp[1]))
        return_dict[metric] = np.mean(rouges)
    return return_dict


def get_matthews_corr(data, predictions):
    # only cola is using this...?
    new_predictions = []
    for prediction in predictions:
        if prediction.strip() == "acceptable":
            new_predictions.append(1.0)
        else:
            new_predictions.append(0.0)
    new_gold = []
    for dp in data:
        if dp[1][0] == "acceptable":
            new_gold.append(1.0)
        else:
            new_gold.append(0.0)
    return matthews_corrcoef(new_gold, new_predictions)


def qa_f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def accuracy(prediction, ground_truth):
    return prediction.lower() == ground_truth.lower()


def get_rouge_over_list(prediction, groundtruth):
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    if len(remove_punc(prediction)) == 0:
        return 0.0
    rouge = Rouge()
    if type(groundtruth) == list:
        if len(groundtruth) == 0:
            return 0
        return np.max([rouge.get_scores(prediction, gt, avg=True)["rouge-l"]["f"] for gt in groundtruth])
    return rouge.get_scores(prediction, groundtruth, avg=True)["rouge-l"]["f"]


def get_accruacy_over_list(prediction, groundtruth):
    if type(groundtruth) == list:
        if len(groundtruth) == 0:
            return 0
        return np.max([accuracy(prediction, gt) for gt in groundtruth])
    return accuracy(prediction, groundtruth)


def get_f1_over_list(prediction, groundtruth):
    if type(groundtruth) == list:
        if len(groundtruth) == 0:
            return 0
        return np.max([qa_f1_score(prediction, gt) for gt in groundtruth])
    return qa_f1_score(prediction, groundtruth)


def get_exact_match_over_list(prediction, groundtruth):
    if type(groundtruth) == list:
        if len(groundtruth) == 0:
            return 0
        return np.max([get_exact_match_over_list(prediction, gt) for gt in groundtruth])
    return (normalize_answer(prediction) == normalize_answer(groundtruth))


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))
