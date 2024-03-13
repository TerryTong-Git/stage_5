'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, f1_score
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
import itertools




class Evaluate_Accuracy(evaluate):
    data = None
    
    def accuracy(self,output, labels):
        preds = output.max(1)[1].type_as(labels)
        correct = preds.eq(labels).double()
        correct = correct.sum()
        return correct / len(labels)
    def recall(self,output, labels):
        preds = output.max(1)[1].type_as(labels)
        true_positive = (preds * labels).sum().double()
        possible_positive = labels.sum().double()
        recall = true_positive / possible_positive if possible_positive != 0 else 0
        return recall

    def precision(self,output, labels):
        preds = output.max(1)[1].type_as(labels)
        true_positive = (preds * labels).sum().double()
        predicted_positive = preds.sum().double()
        precision = true_positive / predicted_positive if predicted_positive != 0 else 0
        return precision

    def f1_score(self,output, labels):
        prec = self.precision(output, labels)
        rec = self.recall(output, labels)
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) != 0 else 0
        return f1

