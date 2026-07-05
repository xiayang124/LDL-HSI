import numpy as np
from sklearn.metrics import confusion_matrix
import torch


class evaluation:
    def __init__(self, targ, pred, label_value=None, binary=None):
        if type(pred) is torch.Tensor:
            pred = pred.cpu().numpy()
            targ = targ.cpu().numpy()
        self.pre = pred
        self.tar = targ
        self.binary = binary
        self.label_value = label_value
        if label_value is None:
            label_value = np.unique(targ)
        self.confusion_matrix = confusion_matrix(targ, pred, labels=label_value)
        
    def _counting_oa(self):
        self.overall_accuracy = np.trace(self.confusion_matrix) / np.sum(self.confusion_matrix) * 100
        return self.overall_accuracy
    
    def _counting_per_class_accuracy(self):
        self.per_class_accuracy = np.zeros(shape=(self.confusion_matrix.shape[0],))
        for classes in range(self.confusion_matrix.shape[0]):
            if np.sum(self.confusion_matrix[classes, :]) != 0:
                self.per_class_accuracy[classes] = self.confusion_matrix[classes, classes] / np.sum(
                    self.confusion_matrix[classes, :]) * 100
            else:
                self.per_class_accuracy[classes] = 0.0
        return self.per_class_accuracy
    
    def _counting_aa(self):
        self.per_class_accuracy = self._counting_per_class_accuracy()
        average_accuracy = np.mean(self.per_class_accuracy)
        return average_accuracy, self.per_class_accuracy
    
    def _counting_kappa(self):
        p0 = np.trace(self.confusion_matrix) / np.sum(self.confusion_matrix)
        pe = np.sum(np.sum(self.confusion_matrix, axis=0) * np.sum(self.confusion_matrix, axis=1)) / (np.sum(self.confusion_matrix) ** 2)
        self.kappa = (p0 - pe) / (1 - pe)
        return self.kappa

    def get_oa(self):
        overall_accuracy = self._counting_oa()
        return overall_accuracy
    
    def get_per_class_accuracy(self):
        per_class_accuracy = self._counting_per_class_accuracy()
        return list(per_class_accuracy)

    def get_aa(self):
        average_accuracy, per_class_accuracy = self._counting_aa()
        return average_accuracy

    def get_kappa(self):
        kappa = self._counting_kappa()
        return kappa * 100
