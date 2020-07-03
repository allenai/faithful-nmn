from typing import Optional, List, Dict
from copy import deepcopy

from overrides import overrides
import numpy as np
import torch
from torchvision.ops.boxes import box_iou

from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric


@Metric.register("classification_module_score")
class ClassificationModuleScore(Metric):
    def __init__(
        self,
        positive_iou_threshold: float = 0.5,
        negative_iou_threshold: float = 0.5,
        recall_thresholds: List[float] = [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
        ],
    ) -> None:
        self.recall_thresholds = recall_thresholds
        self.positive_iou_threshold = positive_iou_threshold
        self.negative_iou_threshold = negative_iou_threshold
        self.cross_entropy = torch.nn.BCELoss(reduction="none")
        # Averaging modes
        # 1: Dataset-wide Score. For every module and for each image to which the module is applied, compute the true positives, true negatives, false positives, false negatives
        # and update the global counts with them. Then use these global counts to compute a single precision, recall, and F1 score for the dataset.
        # 2 (default): Per-instance score. For every module and for each image to which the module is applied, compute the true positives, true negatives, false positives, false negatives
        # and update the instance counts with them. Then use these instance counts to compute a precision, recall, and F1 score for every instance. Dataset score is average
        # over the instance scores.
        # 3: Per-module score. For every module and for each image to which the module is applied, compute a precision, recall, and F1 score. Dataset score is average over the
        # module scores.
        self.averaging_mode = 2
        self.reset()

    def __call__(
        self,
        predictions: List[Dict],
        gold_boxes: List[List[List[float]]],
        proposals: torch.Tensor,
    ):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """
        module_scores = {"non_empty_modules": []}
        n_objects = proposals.shape[1]
        all_ious = {}
        instance_precision_good = {"overall": 0}
        instance_precision_bad = {"overall": 0}
        instance_recall_good = {"overall": 0}
        instance_recall_bad = {"overall": 0}
        for pred_module in predictions:
            if pred_module["module"] == "count":
                continue
            ious1 = box_iou(
                proposals[0, :, :].cpu(),
                torch.from_numpy(np.array(gold_boxes[pred_module["index"]][0]))
                .float()
                .view(-1, 4),
            )
            ious2 = box_iou(
                proposals[1, :, :].cpu(),
                torch.from_numpy(np.array(gold_boxes[pred_module["index"]][1]))
                .float()
                .view(-1, 4),
            )
            has_gold_boxes = [True, True]
            if len(gold_boxes[pred_module["index"]][0]) == 0:
                has_gold_boxes[0] = False
                ious1 = torch.zeros((ious1.shape[0], 1))
            if len(gold_boxes[pred_module["index"]][1]) == 0:
                has_gold_boxes[1] = False
                ious2 = torch.zeros((ious2.shape[0], 1))
            ious = [ious1.numpy(), ious2.numpy()]
            all_ious[pred_module["index"]] = [ious[0].tolist(), ious[1].tolist()]
            scores = [0, 0]
            module_scores[pred_module["index"]] = [0, 0]
            for img in range(2):
                module_scores["non_empty_modules"].append((pred_module["index"], img))
                recall_good = 0
                recall_bad = 0
                precision_good = 0
                precision_bad = 0
                if has_gold_boxes[img]:
                    gold_indices = set()
                    predictions = []
                    for i in range(ious[img].shape[1]):
                        max_score = 0
                        this_match = set()
                        for j in range(ious[img].shape[0]):
                            if ious[img][j, i] > self.positive_iou_threshold:
                                gold_indices.add(j)
                                this_match.add(j)
                                max_score = max(
                                    pred_module["output"][img * n_objects + j].item(),
                                    max_score,
                                )
                        if max_score > 0.5:
                            recall_good += 1
                        else:
                            recall_bad += 1
                for j in range(ious[img].shape[0]):
                    if pred_module["output"][img * n_objects + j].item() > 0.5:
                        if ious[img][j, :].max().item() > self.positive_iou_threshold:
                            precision_good += 1
                        elif ious[img][j, :].max().item() < self.negative_iou_threshold:
                            precision_bad += 1
                if precision_good + precision_bad > 0:
                    precision = precision_good / float(precision_good + precision_bad)
                else:
                    precision = 1.0
                if recall_good + recall_bad > 0:
                    recall = recall_good / float(recall_good + recall_bad)
                else:
                    recall = 1.0
                score = self.calc_f1(precision, recall)
                module_scores[pred_module["index"]][img] = score
                instance_precision_good["overall"] += precision_good
                instance_precision_bad["overall"] += precision_bad
                instance_recall_good["overall"] += recall_good
                instance_recall_bad["overall"] += recall_bad
                if self.averaging_mode == 1:
                    self.precision_good[pred_module["module"]] += precision_good
                    self.precision_bad[pred_module["module"]] += precision_bad
                    self.recall_good[pred_module["module"]] += recall_good
                    self.recall_bad[pred_module["module"]] += recall_bad
                if pred_module["module"] not in instance_precision_good:
                    instance_precision_good[pred_module["module"]] = 0
                    instance_precision_bad[pred_module["module"]] = 0
                    instance_recall_good[pred_module["module"]] = 0
                    instance_recall_bad[pred_module["module"]] = 0
                instance_precision_good[pred_module["module"]] += precision_good
                instance_precision_bad[pred_module["module"]] += precision_bad
                instance_recall_good[pred_module["module"]] += recall_good
                instance_recall_bad[pred_module["module"]] += recall_bad
                if self.averaging_mode == 3:
                    self.f1_scores[pred_module["module"]].append(score)
                    self.precisions[pred_module["module"]].append(precision)
                    self.recalls[pred_module["module"]].append(recall)
        instance_scores = {}
        for key in instance_precision_good:
            if instance_precision_good[key] + instance_precision_bad[key] > 0:
                instance_precision = instance_precision_good[key] / float(
                    instance_precision_good[key] + instance_precision_bad[key]
                )
            else:
                instance_precision = 1.0
            if instance_recall_good[key] + instance_recall_bad[key] > 0:
                instance_recall = instance_recall_good[key] / float(
                    instance_recall_good[key] + instance_recall_bad[key]
                )
            else:
                instance_recall = 1.0
            f1 = self.calc_f1(instance_precision, instance_recall)
            instance_scores[key] = f1
            if self.averaging_mode == 2:
                self.f1_scores[key].append(f1)
                self.precisions[key].append(instance_precision)
                self.recalls[key].append(instance_recall)
        if self.averaging_mode == 1:
            return (
                module_scores,
                [
                    instance_precision_good["overall"],
                    instance_precision_bad["overall"],
                    instance_recall_good["overall"],
                    instance_recall_bad["overall"],
                ],
            )
        else:
            return module_scores, instance_scores["overall"]

    def compute_best_box_predictions(self, predictions, gold_boxes, proposals):
        best_predictions = deepcopy(predictions)
        for index, pred_module in enumerate(predictions):
            if pred_module["module"] == "count":
                continue
            ious1 = box_iou(
                proposals[0, :, :].cpu(),
                torch.from_numpy(np.array(gold_boxes[pred_module["index"]][0]))
                .float()
                .view(-1, 4),
            )
            ious2 = box_iou(
                proposals[1, :, :].cpu(),
                torch.from_numpy(np.array(gold_boxes[pred_module["index"]][1]))
                .float()
                .view(-1, 4),
            )
            has_gold_boxes = [True, True]
            if len(gold_boxes[pred_module["index"]][0]) == 0:
                has_gold_boxes[0] = False
                ious1 = torch.zeros((ious1.shape[0], 1))
            if len(gold_boxes[pred_module["index"]][1]) == 0:
                has_gold_boxes[1] = False
                ious2 = torch.zeros((ious2.shape[0], 1))
            ious = [ious1.numpy(), ious2.numpy()]
            probabilities = []
            for img in range(2):
                if has_gold_boxes[img]:
                    for i in range(ious[img].shape[0]):
                        if ious[img][i, :].max().item() > self.positive_iou_threshold:
                            probabilities.append(1.0)
                        else:
                            probabilities.append(0.0)
                else:
                    probabilities += [0.0 for _ in range(ious[img].shape[0])]
                best_predictions[index]["output"] = torch.Tensor(probabilities).to(
                    pred_module["output"].device
                )
        return best_predictions

    def calc_f1(self, precision, recall):
        if max(precision, recall) == 0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)

    def get_metric(self, reset: bool = False, module: str = None):
        """
        Returns
        -------
        The accumulated accuracy.
        """
        score = {"recall": 0.0, "precision": 0.0, "f1": 0.0}
        if module is not None:
            if self.averaging_mode in {2, 3}:
                if len(self.f1_scores[module]) > 0:
                    score["f1"] = np.mean(self.f1_scores[module])
                score["precision"] = np.mean(self.precisions[module])
                score["recall"] = np.mean(self.recalls[module])
            else:
                precision = 1.0
                if self.precision_good[module] + self.precision_bad[module] > 0:
                    precision = self.precision_good[module] / float(
                        self.precision_good[module] + self.precision_bad[module]
                    )
                recall = 1.0
                if self.recall_good[module] + self.recall_bad[module] > 0:
                    recall = self.recall_good[module] / float(
                        self.recall_good[module] + self.recall_bad[module]
                    )
                score["f1"] = self.calc_f1(precision, recall)
                score["precision"] = precision
                score["recall"] = recall
        elif self.averaging_mode == 2 and len(self.f1_scores["overall"]) > 0:
            score["f1"] = np.mean(self.f1_scores["overall"])
            score["precision"] = np.mean(self.precisions["overall"])
            score["recall"] = np.mean(self.recalls["overall"])
        elif self.averaging_mode == 3 and any(
            [len(self.f1_scores[m]) > 0 for m in self.modules]
        ):
            all_f1_scores = [s for m in self.f1_scores for s in self.f1_scores[m]]
            score["f1"] = np.mean(all_f1_scores)
            all_precisions = [s for m in self.precisions for s in self.precisions[m]]
            score["precision"] = np.mean(all_precisions)
            all_recalls = [s for m in self.recalls for s in self.recalls[m]]
            score["recall"] = np.mean(all_recalls)
        elif self.averaging_mode == 1 and any(
            [
                self.precision_good[m]
                + self.precision_bad[m]
                + self.recall_good[m]
                + self.recall_bad[m]
                > 0
                for m in self.modules
            ]
        ):
            precision_good = sum([self.precision_good[m] for m in self.precision_good])
            precision_bad = sum([self.precision_bad[m] for m in self.precision_bad])
            recall_good = sum([self.recall_good[m] for m in self.recall_good])
            recall_bad = sum([self.recall_bad[m] for m in self.recall_bad])
            precision = 1.0
            if precision_good + precision_bad > 0:
                precision = precision_good / float(precision_good + precision_bad)
            recall = 1.0
            if recall_good + recall_bad > 0:
                recall = recall_good / float(recall_good + recall_bad)
            score["f1"] = self.calc_f1(precision, recall)
            score["precision"] = precision
            score["recall"] = recall
        if reset:
            self.reset()
        return score

    @overrides
    def reset(self):
        self.modules = ["find", "filter", "with_relation", "project"]
        self.precision_good = {m: 0 for m in self.modules}
        self.precision_bad = {m: 0 for m in self.modules}
        self.recall_good = {m: 0 for m in self.modules}
        self.recall_bad = {m: 0 for m in self.modules}
        self.precisions = {m: [] for m in self.modules}
        self.recalls = {m: [] for m in self.modules}
        self.f1_scores = {m: [] for m in self.modules}
        if self.averaging_mode == 2:
            self.f1_scores["overall"] = []
            self.precisions["overall"] = []
            self.recalls["overall"] = []
