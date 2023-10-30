from allennlp.predictors import Predictor
import streamlit as st
import textattack
from textattack.attack_results import  FailedAttackResult,MaximizedAttackResult,SkippedAttackResult,SuccessfulAttackResult
import matplotlib.pyplot as plt
from textattack.metrics.attack_metrics import (
    AttackQueries,
    AttackSuccessRate,
    WordsPerturbed,
)
# from IPython.core.display import display, HTML
class AllenNLPModel(textattack.models.wrappers.ModelWrapper):
    def __init__(self):
        self.predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/basic_stanford_sentiment_treebank-2020.06.09.tar.gz")
        self.model = self.predictor._model
        self.tokenizer = self.predictor._dataset_reader._tokenizer

    def __call__(self, text_input_list):
        outputs = []
        for text_input in text_input_list:
            outputs.append(self.predictor.predict(sentence=text_input))
        # For each output, outputs['logits'] contains the logits where
        # index 0 corresponds to the positive and index 1 corresponds
        # to the negative score. We reverse the outputs (by reverse slicing,
        # [::-1]) so that negative comes first and positive comes second.
        return [output['logits'][::-1] for output in outputs]

model_wrapper = AllenNLPModel()
from textattack.datasets import HuggingFaceDataset
from textattack.attack_recipes import TextBuggerLi2018
from textattack.attacker import Attacker

st.title('TextAttack Visualization')
dataset = HuggingFaceDataset("glue", "sst2", "train")
attack = TextBuggerLi2018.build(model_wrapper)

attacker = Attacker(attack, dataset)
x=attacker.attack_dataset()
num_skipped=0
num_successes=0
num_failures=0
for result in x:
    if isinstance(result, SkippedAttackResult):
        num_skipped += 1
    if isinstance(result, (SuccessfulAttackResult, MaximizedAttackResult)):
        num_successes += 1
    if isinstance(result, FailedAttackResult):
        num_failures += 1
print("Skipped:"+str(num_skipped))
print("Success:"+str(num_successes))
print("Failure:"+str(num_failures))

labels = ["Success", "Failure", "Skipped"]
sizes = [num_successes, num_failures, num_skipped]
colors = ["green", "red", "gray"]
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
st.pyplot(plt)
print(len(x))
attack_success_stats = AttackSuccessRate().calculate(x)
words_perturbed_stats = WordsPerturbed().calculate(x)
attack_query_stats = AttackQueries().calculate(x)
original_accuracy=str(attack_success_stats["original_accuracy"])
attack_accuracy_perc=str(attack_success_stats["attack_accuracy_perc"])
attack_success_rate=str(attack_success_stats["attack_success_rate"])
avg_word_perturbed_perc=str(words_perturbed_stats["avg_word_perturbed_perc"])
avg_word_perturbed=str(words_perturbed_stats["avg_word_perturbed"])
avg_num_queries=attack_query_stats["avg_num_queries"]


print("original_accuracy:"+ str(attack_success_stats["original_accuracy"]) + "%")
print("attack_accuracy_perc:"+ str(attack_success_stats["attack_accuracy_perc"]) + "%")
print("attack_success_rate:"+  str(attack_success_stats["attack_success_rate"]) + "%")
print("avg_word_perturbed_perc:"+  str(words_perturbed_stats["avg_word_perturbed_perc"]) + "%")
print("avg_word_perturbed:"+  str(words_perturbed_stats["avg_word_perturbed"]))
print("Avg num queries:", attack_query_stats["avg_num_queries"])
st.bar_chart({"original_accuracy": original_accuracy, "attack_accuracy_perc": attack_accuracy_perc, "attack_success_rate": attack_success_rate,"avg_word_perturbed_perc":avg_word_perturbed_perc})

st.bar_chart({"avg_word_perturbed":avg_word_perturbed,"avg_num_queries":avg_num_queries})
