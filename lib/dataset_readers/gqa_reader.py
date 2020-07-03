from glob import glob
from typing import Dict, List
import json
import re
import os
import sys
import pickle
import bisect
from copy import copy

from overrides import overrides
import numpy as np
import torch
from torchvision.ops.boxes import box_iou

# from spacy_pytorch_transformers.pipeline.wordpiecer import align_word_pieces
from spacy_transformers.pipeline.wordpiecer import align_word_pieces

from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (
    ArrayField,
    LabelField,
    TextField,
    MetadataField,
    ProductionRuleField,
    ListField,
    IndexField,
)
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import (
    TokenIndexer,
    SingleIdTokenIndexer,
    PretrainedTransformerIndexer,
)
from allennlp.data.tokenizers import (
    Tokenizer,
    WordTokenizer,
    PretrainedTransformerTokenizer,
)

from lib.semparse.visual_reasoning_gqa_language import VisualReasoningGqaLanguage
from lib.dataset_readers.reader_utils import target_sequence_to_target_attn
from lib.modules.seq2seq_encoders.lxmert_src.utils import load_obj_tsv


@DatasetReader.register("gqa")
class GqaReader(DatasetReader):
    def __init__(
        self,
        image_feat_path: str,
        topk: int = -1,
        lazy: bool = False,
        reload_tsv: bool = False,
        cache_path: str = "",
        relations_path: str = "",
        attributes_path: str = "",
        objects_path: str = "",
        positive_threshold: float = 0.5,
        negative_threshold: float = 0.5,
        object_supervision: bool = False,
        require_some_positive: bool = False,
    ) -> None:
        super().__init__(lazy)
        self._tokenizer = PretrainedTransformerTokenizer(
            "bert-base-uncased", do_lowercase=True
        )
        self._word_tokenizer = WordTokenizer()
        self._token_indexers = {
            "tokens": PretrainedTransformerIndexer(
                "bert-base-uncased", do_lowercase=True
            )
        }

        self._language = VisualReasoningGqaLanguage(None, None, None, None, None)
        self._production_rules = self._language.all_possible_productions()
        self._action_map = {rule: i for i, rule in enumerate(self._production_rules)}
        production_rule_fields = [
            ProductionRuleField(rule, is_global_rule=True)
            for rule in self._production_rules
        ]
        self._production_rule_field = ListField(production_rule_fields)
        self._image_feat_cache_dir = os.path.join(
            "cache", image_feat_path.split("/")[-1]
        )
        if len(cache_path) > 0:
            self._image_feat_cache_dir = os.path.join(
                cache_path, "cache", image_feat_path.split("/")[-1]
            )
        self.img_data = None
        if reload_tsv:
            self.img_data = load_obj_tsv(
                image_feat_path,
                topk,
                save_cache=False,
                cache_path=self._image_feat_cache_dir,
            )
            self.img_data = {img["img_id"]: img for img in self.img_data}
        self.object_data = None
        self.attribute_data = None
        if len(objects_path) > 0:
            self.object_data = json.load(open(objects_path))
            self.object_data = {img["image_id"]: img for img in self.object_data}
        if len(attributes_path) > 0:
            self.attribute_data = json.load(open(attributes_path))
            self.attribute_data = {img["image_id"]: img for img in self.attribute_data}
        if len(relations_path) > 0:
            self.relation_data = json.load(open(relations_path))
            self.relation_data = {img["image_id"]: img for img in self.relation_data}
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.object_supervision = object_supervision
        self.require_some_positive = require_some_positive

    def action_sequence_to_logical_form(self, action_sequence):
        if len(action_sequence) == 0:
            return "", []
        elif len(action_sequence) == 1:
            return action_sequence[0], []
        action = action_sequence[0]
        if action in {"and", "or"}:
            action = "bool_" + action
        action_sequence = action_sequence[1:]
        num_args = 0
        if action != "find":
            num_args += 1
        if action in {"with_relation", "bool_and", "bool_or"}:
            num_args += 1
        if num_args == 1:
            arg, action_sequence = self.action_sequence_to_logical_form(action_sequence)
            return "(" + action + " " + arg + ")", action_sequence
        elif num_args == 2:
            arg, action_sequence = self.action_sequence_to_logical_form(action_sequence)
            arg2, action_sequence = self.action_sequence_to_logical_form(
                action_sequence
            )
            return "(" + action + " " + arg + " " + arg2 + ")", action_sequence
        return action, action_sequence

    def assign_attention_to_tokens(self, gold_attentions, sentence):
        sentence = sentence.lower()
        tokens = [str(tok) for tok in self._word_tokenizer.tokenize(sentence)]
        token_starts = []
        token_ends = []
        suffix = copy(sentence)
        suffix_start_ind = 0
        for tok in tokens:
            start = suffix.find(tok)
            if start == -1:
                assert False
            token_starts.append(start + suffix_start_ind)
            suffix_start_ind += start + len(tok)
            token_ends.append(suffix_start_ind)
            suffix = suffix[start + len(tok) :]
        wordpieces = [str(tok).strip("#") for tok in self._tokenizer.tokenize(sentence)]
        alignment = align_word_pieces(tokens, wordpieces[1:-1])
        for i in range(len(alignment)):
            alignment[i] = [j + 1 for j in alignment[i]]
        assert alignment is not None
        gold_numeric_attentions = []
        for att in gold_attentions:
            attention = np.zeros((len(wordpieces)))
            match_flag = False
            att = att.lower()
            for match in re.finditer(att, sentence):
                char_start, char_end = match.span()
                start_token_index = bisect.bisect_left(token_starts, char_start)
                end_token_index = bisect.bisect_left(token_ends, char_end)
                if (
                    token_starts[start_token_index] == char_start
                    and token_ends[end_token_index] == char_end
                ):
                    selected_wp = []
                    for j in range(start_token_index, end_token_index + 1):
                        attention[alignment[j]] = 1
                        selected_wp += wordpieces[
                            alignment[j][0] : alignment[j][-1] + 1
                        ]
                    match_flag = True
                elif (
                    token_ends[end_token_index] == char_end + 1
                    and sentence[token_ends[end_token_index] - 1] == "s"
                ):
                    selected_wp = []
                    for j in range(start_token_index, end_token_index + 1):
                        attention[alignment[j]] = 1
                        selected_wp += wordpieces[
                            alignment[j][0] : alignment[j][-1] + 1
                        ]
                    match_flag = True
            """if not match_flag:
                print(att, sentence, tokens, wordpieces, alignment, token_starts, token_ends)"""
            gold_numeric_attentions.append(attention)
        return np.stack(gold_numeric_attentions)

    def target_sequence_to_target_objs(self, target_sequence, objs) -> List:
        target_objs = [[] for _ in target_sequence]
        module_obj_i = 0
        for action_i, action in enumerate(target_sequence):
            rhs = action.split(" -> ")[1]
            if module_obj_i < len(objs):
                module_name, obj = objs[module_obj_i]
                if rhs == module_name:
                    target_objs[action_i] = obj
                    module_obj_i += 1
        return target_objs

    @overrides
    def _read(self, file_path: str):
        with open(file_path) as f:
            lines = f.readlines()
            examples = [json.loads(line) for line in lines]
            for e in examples:
                if (
                    self.img_data is not None and e["imageId"] not in self.img_data
                ) or (
                    self.img_data is None
                    and not os.path.exists(
                        os.path.join(self._image_feat_cache_dir, e["imageId"])
                    )
                ):
                    continue
                action_sequence = []
                actions_with_attentions = []
                gold_attentions = []
                objects = []
                objects_ops = []
                children = []
                for op in e["semantic"]:
                    if op["operation"] == "relate":
                        subj_obj_type = op["argument"].split(",")[2][0]
                        obj_name = op["argument"].split(",")[0]
                        predicate = op["argument"].split(" (")[0].split(",")[1]
                        if subj_obj_type == "s":
                            action_sequence.append("find")
                            actions_with_attentions.append("find")
                            gold_attentions.append(obj_name)
                            if self.object_data is not None:
                                gold_objects = []
                                if "(" in op["argument"]:
                                    id_string = (
                                        op["argument"].split("(")[1].split(")")[0]
                                    )
                                    if id_string != "-" and id_string.isnumeric():
                                        obj_id = int(id_string)
                                        img = self.object_data[int(e["imageId"])]
                                        for obj in img["objects"]:
                                            if obj_id == obj["object_id"]:
                                                obj1 = obj
                                                break
                                        gold_objects.append(
                                            [
                                                obj1["x"],
                                                obj1["y"],
                                                obj1["x"] + obj1["w"],
                                                obj1["y"] + obj1["h"],
                                            ]
                                        )
                                        for obj in img["objects"]:
                                            if (
                                                obj_id != obj["object_id"]
                                                and "synsets" in obj1
                                                and "synsets" in obj
                                                and len(obj1["synsets"]) > 0
                                                and len(obj["synsets"]) > 0
                                                and obj1["synsets"][0]
                                                == obj["synsets"][0]
                                            ):
                                                gold_objects.append(
                                                    [
                                                        obj["x"],
                                                        obj["y"],
                                                        obj["x"] + obj["w"],
                                                        obj["y"] + obj["h"],
                                                    ]
                                                )
                                objects.append(gold_objects)
                                objects_ops.append("find")
                        else:
                            find_index = len(action_sequence) - 1
                            num_actions_with_attention = 1
                            child_tracking = children[find_index]
                            while find_index >= 0 and child_tracking > 0:
                                if action_sequence[find_index] in {
                                    "find",
                                    "filter",
                                    "with_relation",
                                }:
                                    num_actions_with_attention += 1
                                if children[find_index] == 0:
                                    child_tracking -= 1
                                    if child_tracking == 0:
                                        break
                                elif children[find_index] > child_tracking:
                                    child_tracking = children[find_index]
                                find_index -= 1
                            assert find_index >= 0
                            action_sequence = (
                                action_sequence[:find_index]
                                + ["find"]
                                + action_sequence[find_index:]
                            )
                            children = (
                                children[:find_index]
                                + [0]
                                + action_sequence[find_index:]
                            )
                            actions_with_attentions = (
                                actions_with_attentions[:-num_actions_with_attention]
                                + ["find"]
                                + actions_with_attentions[-num_actions_with_attention:]
                            )
                            gold_attentions.append(obj_name)
                            if self.object_data is not None:
                                gold_objects = []
                                if "(" in op["argument"]:
                                    id_string = (
                                        op["argument"].split("(")[1].split(")")[0]
                                    )
                                    if id_string != "-" and id_string.isnumeric():
                                        obj_id = int(id_string)
                                        img = self.object_data[int(e["imageId"])]
                                        for obj in img["objects"]:
                                            if obj_id == obj["object_id"]:
                                                obj1 = obj
                                                break
                                        gold_objects.append(
                                            [
                                                obj1["x"],
                                                obj1["y"],
                                                obj1["x"] + obj1["w"],
                                                obj1["y"] + obj1["h"],
                                            ]
                                        )
                                        for obj in img["objects"]:
                                            if (
                                                obj_id != obj["object_id"]
                                                and "synsets" in obj1
                                                and "synsets" in obj
                                                and len(obj1["synsets"]) > 0
                                                and len(obj["synsets"]) > 0
                                                and obj1["synsets"][0]
                                                == obj["synsets"][0]
                                            ):
                                                gold_objects.append(
                                                    [
                                                        obj["x"],
                                                        obj["y"],
                                                        obj["x"] + obj["w"],
                                                        obj["y"] + obj["h"],
                                                    ]
                                                )
                                objects = (
                                    objects[:-num_actions_with_attention]
                                    + [gold_objects]
                                    + objects[-num_actions_with_attention:]
                                )
                                objects_ops = (
                                    objects_ops[:-num_actions_with_attention]
                                    + ["find"]
                                    + objects_ops[-num_actions_with_attention:]
                                )
                        action_sequence.append("with_relation")
                        actions_with_attentions.append("with_relation")
                        children.append(2)
                        gold_attentions.append(predicate)
                        gold_objects = []
                        if "(" in op["argument"]:
                            id_string = op["argument"].split("(")[1].split(")")[0]
                            if id_string != "-" and id_string.isnumeric():
                                obj_id = int(id_string)
                                img = self.relation_data[int(e["imageId"])]
                                for rel in img["relationships"]:
                                    if subj_obj_type == "s" and obj_id == int(
                                        rel["subject"]["object_id"]
                                    ):
                                        gold_objects.append(
                                            [
                                                rel["subject"]["x"],
                                                rel["subject"]["y"],
                                                rel["subject"]["x"]
                                                + rel["subject"]["w"],
                                                rel["subject"]["y"]
                                                + rel["subject"]["h"],
                                            ]
                                        )
                                    elif subj_obj_type == "o" and obj_id == int(
                                        rel["object"]["object_id"]
                                    ):
                                        gold_objects.append(
                                            [
                                                rel["object"]["x"],
                                                rel["object"]["y"],
                                                rel["object"]["x"] + rel["object"]["w"],
                                                rel["object"]["y"] + rel["object"]["h"],
                                            ]
                                        )
                        objects.append(gold_objects)
                        objects_ops.append("with_relation")
                    elif op["operation"] == "select":
                        action_sequence.append("find")
                        actions_with_attentions.append("find")
                        children.append(0)
                        gold_attentions.append(op["argument"].split(" (")[0])
                        if self.object_data is None:
                            continue
                        gold_objects = []
                        if "(" in op["argument"]:
                            id_string = op["argument"].split("(")[1].split(")")[0]
                            if id_string != "-" and id_string.isnumeric():
                                obj_id = int(id_string)
                                img = self.object_data[int(e["imageId"])]
                                for obj in img["objects"]:
                                    if obj_id == obj["object_id"]:
                                        obj1 = obj
                                        break
                                gold_objects.append(
                                    [
                                        obj1["x"],
                                        obj1["y"],
                                        obj1["x"] + obj1["w"],
                                        obj1["y"] + obj1["h"],
                                    ]
                                )
                                for obj in img["objects"]:
                                    if (
                                        obj_id != obj["object_id"]
                                        and "synsets" in obj1
                                        and "synsets" in obj
                                        and len(obj1["synsets"]) > 0
                                        and len(obj["synsets"]) > 0
                                        and obj1["synsets"][0] == obj["synsets"][0]
                                    ):
                                        gold_objects.append(
                                            [
                                                obj["x"],
                                                obj["y"],
                                                obj["x"] + obj["w"],
                                                obj["y"] + obj["h"],
                                            ]
                                        )
                        objects.append(gold_objects)
                        objects_ops.append("find")
                    elif op["operation"] == "number":
                        action_sequence.append(str(op["argument"]))
                    elif op["operation"].split()[0] == "filter":
                        action_sequence.append("filter")
                        actions_with_attentions.append("filter")
                        children.append(1)
                        if "(" in op["argument"]:
                            gold_attentions.append(
                                op["argument"].split("(")[0]
                                + " "
                                + op["argument"].split("(")[1].split(")")[0]
                            )
                        else:
                            gold_attentions.append(op["argument"])
                        if self.attribute_data is None or self.object_data is None:
                            continue
                        img = self.attribute_data[int(e["imageId"])]
                        negate = False
                        attr = op["argument"]
                        if "(" in op["argument"]:
                            negate = True
                            attr = op["argument"].split("(")[1].split(")")[0]
                        attr = attr.lower()
                        gold_objects = []
                        for obj in img["attributes"]:
                            obj_attrs = []
                            if "attributes" in obj:
                                obj_attrs = [a.lower() for a in obj["attributes"]]
                            if attr in obj_attrs and not negate:
                                gold_objects.append(
                                    [
                                        obj["x"],
                                        obj["y"],
                                        obj["x"] + obj["w"],
                                        obj["y"] + obj["h"],
                                    ]
                                )
                            elif attr not in obj_attrs and negate:
                                gold_objects.append(
                                    [
                                        obj["x"],
                                        obj["y"],
                                        obj["x"] + obj["w"],
                                        obj["y"] + obj["h"],
                                    ]
                                )
                        objects.append(gold_objects)
                        objects_ops.append("filter")
                    else:
                        action_sequence.append(op["operation"].split()[0])
                assert len(actions_with_attentions) == len(gold_attentions)
                logical_form, _ = self.action_sequence_to_logical_form(
                    action_sequence[::-1]
                )
                gold_question_attentions = self.assign_attention_to_tokens(
                    gold_attentions, e["question"].lower()
                )
                module_attn = [
                    (act, att)
                    for act, att in zip(
                        actions_with_attentions, gold_question_attentions
                    )
                ]
                module_attn = module_attn[::-1]
                formal_action_sequence = self._language.logical_form_to_action_sequence(
                    logical_form
                )
                gold_question_attentions = target_sequence_to_target_attn(
                    formal_action_sequence, module_attn, full_array=True
                )
                gold_question_attentions = np.stack(gold_question_attentions)
                find_object_pairs = [(op, obj) for obj, op in zip(objects, objects_ops)]
                find_object_pairs = find_object_pairs[::-1]
                find_matched_objects = self.target_sequence_to_target_objs(
                    formal_action_sequence, find_object_pairs
                )
                instance = self.text_to_instance(
                    e["question"],
                    e["imageId"],
                    logical_form,
                    gold_question_attentions,
                    find_matched_objects,
                    e["answer"],
                )
                if instance is not None:
                    yield instance

    @overrides
    def text_to_instance(
        self,  # type: ignore
        question: str,
        image_id: str,
        logical_form: str = None,
        gold_question_attentions: np.ndarray = None,
        objects: List[List[int]] = None,
        denotation: str = None,
    ) -> Instance:
        tokenized_sentence = self._tokenizer.tokenize(question.lower())
        sentence_field = TextField(tokenized_sentence, self._token_indexers)
        if self.img_data is not None:
            img_info = self.img_data[image_id]
        else:
            img_info = pickle.load(
                open(os.path.join(self._image_feat_cache_dir, image_id), "rb")
            )
        visual_feat = img_info["features"].copy()
        boxes = img_info["boxes"].copy()
        fields = {
            "visual_feat": ArrayField(visual_feat),
            "pos": ArrayField(boxes),
            "question_field": sentence_field,
            "image_id": MetadataField(image_id),
            "actions": self._production_rule_field,
        }

        if denotation is not None:
            if denotation.lower() in {"yes", "true"}:
                fields["denotation"] = ArrayField(np.array(1))
            else:
                fields["denotation"] = ArrayField(np.array(0))
        if logical_form is not None:
            fields["logical_form"] = MetadataField(logical_form)
            actions = self._language.logical_form_to_action_sequence(logical_form)
            index_fields = []
            for production_rule in actions:
                index_fields.append(
                    IndexField(
                        self._action_map[production_rule], self._production_rule_field
                    )
                )
            fields["target_action_sequence"] = ListField(index_fields)
        if gold_question_attentions is not None:
            fields["gold_question_attentions"] = ArrayField(gold_question_attentions)
        if objects is not None and self.object_supervision:
            proposals = torch.from_numpy(boxes)
            gold_proposal_choices = np.zeros((len(objects), boxes.shape[0]))
            for i in range(len(objects)):
                if len(objects[i]) > 0:
                    ious = box_iou(
                        torch.from_numpy(np.array(objects[i])).float(),
                        proposals.float(),
                    )
                    ious = ious.numpy()
                    for j in range(ious.shape[0]):
                        if ious[j].max() > 0:
                            gold_proposal_choices[
                                i, ious[j, :] >= self.positive_threshold
                            ] = 1
                            gold_proposal_choices[
                                i, ious[j, :] < self.negative_threshold
                            ] = -1
            fields["gold_object_choices"] = ArrayField(gold_proposal_choices)
            if gold_proposal_choices.max() <= 0 and self.require_some_positive:
                return None
            return Instance(fields)
        if self.object_supervision:
            return None
        return Instance(fields)
