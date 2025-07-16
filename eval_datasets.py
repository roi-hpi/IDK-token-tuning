import random
from csv import reader

from datasets import load_dataset

from utils import read_from_csv


class QABenchmark:
    def __init__(self):
        self.dataset = []

    def sample(self, k: int):
        return random.sample(self.dataset, min(k, len(self.dataset)))


class Squad(QABenchmark):
    def __init__(self):
        super().__init__()
        loaded_dataset = load_dataset("squad", split="validation")
        self.dataset = [(example["question"], example["answers"]["text"][0]) for example in loaded_dataset]


class LamaTrex(QABenchmark):
    def __init__(self, split: str = "train"):
        super().__init__()
        loaded_dataset = load_dataset("lama", split=split)
        print(loaded_dataset)
        self.dataset = [
            (example["masked_sentence"][:-7], example["obj_label"])
            for example in loaded_dataset
            if example["masked_sentence"][-7:] == "[MASK]."
        ]


class LamaSqaud(QABenchmark):
    def __init__(self):
        super().__init__()
        loaded_dataset = load_dataset("lama", "squad")
        self.dataset = [
            (example["masked_sentence"][:-7], example["obj_label"])
            for example in loaded_dataset
            if example["masked_sentence"][-7:] == "[MASK]."
        ]


class LamaGoogleRE(QABenchmark):
    def __init__(self):
        super().__init__()
        loaded_dataset = load_dataset("lama", "google_re")
        self.dataset = [
            (example["masked_sentence"][:-7], example["obj_label"])
            for example in loaded_dataset
            if example["masked_sentence"][-7:] == "[MASK]."
        ]


class TriviaQA(QABenchmark):
    def __init__(self, split="validation"):
        super().__init__()
        loaded_dataset = load_dataset("trivia_qa", "rc", split=split)
        instruction = "Please answer the following question: "
        self.dataset = [
            (instruction + example["question"], list(set([example["answer"]["value"]] + example["answer"]["aliases"])))
            for example in loaded_dataset
        ]


class NaturalQuestions(QABenchmark):
    def __init__(self, split="validation"):
        super().__init__()
        loaded_dataset = load_dataset("cjlovering/natural-questions-short", split=split)
        self.dataset = [
            (example["questions"][0]["input_text"], example["answers"][0]["span_text"]) for example in loaded_dataset
        ]


class PopQA(QABenchmark):
    def __init__(self, split="test"):
        super().__init__()
        loaded_dataset = load_dataset("akariasai/PopQA", split=split)
        instruction = "Please answer the following question: "

        self.dataset = [
            (instruction + example["question"], example["possible_answers"].strip("][").split(", "))
            for example in loaded_dataset
        ]


class FakeFacts(QABenchmark):
    def __init__(self, csv_path: str):
        super().__init__()
        csv_list = read_from_csv(csv_path)
        self.dataset = [(example[0], True if example[1] == "TRUE" else False) for example in csv_list]


class HotpotQA(QABenchmark):
    def __init__(self, split="validation"):
        super().__init__()
        loaded_dataset = load_dataset("hotpot_qa", "fullwiki", split=split)
        self.dataset = [(example["question"], example["answer"]) for example in loaded_dataset]


class CommonSenseQA(QABenchmark):
    def __init__(self, split="validation"):
        super().__init__()
        loaded_dataset = load_dataset("commonsense_qa", split=split)
        self.dataset = []
        for example in loaded_dataset:
            question = example["question"]
            labels = example["choices"]["label"]
            answers = example["choices"]["text"]
            right_answer_key = example["answerKey"]
            right_answer_ind = labels.index(right_answer_key)
            right_answer = answers[right_answer_ind]
            self.dataset.append((question, right_answer))


# class ourSmallDataSets(QABenchmark):#talk to roi on how to fix this
#    def __init__(self):
#        super().__init__()
#        with open('./data/triviaQA_truthfulness/known.csv', 'r') as known:
#            csv_reader = reader(known)
#            for row in csv_reader:
