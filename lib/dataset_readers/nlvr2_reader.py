import csv
import json
import logging
import os
import pickle
from copy import copy
from typing import Dict, List, Any

import numpy as np

from allennlp.data import Field
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (
    TextField,
    ProductionRuleField,
    MetadataField,
    ListField,
    IndexField,
    ArrayField,
)
from allennlp.data.fields.label_field import LabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import PretrainedTransformerTokenizer, WordTokenizer
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.semparse import ParsingError
from allennlp.common.file_utils import cached_path
from overrides import overrides

# from spacy_pytorch_transformers.pipeline.wordpiecer import align_word_pieces
from spacy_transformers.pipeline.wordpiecer import align_word_pieces

from lib.dataset_readers.reader_utils import (
    annotation_to_lisp_exp,
    annotation_to_module_attention,
    target_sequence_to_target_attn,
    target_sequence_to_target_boxes,
)
from lib.modules.seq2seq_encoders.lxmert_src.utils import load_obj_tsv
from lib.semparse.visual_reasoning_nlvr2_language import VisualReasoningNlvr2Language

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("nlvr2")
class Nlvr2NMNDatasetReader(DatasetReader):
    def __init__(
        self,
        image_feat_path: str,
        annotations_path: str,
        box_annotations_path: str = None,
        lazy: bool = False,
        tiny: bool = False,
        max_boxes: int = 36,
        max_seq_length: int = -1,
        cache_path: str = "",
        only_with_annotation: bool = False,
        only_with_box_annotation: bool = False,
        simple_exist_annotation: bool = False,
        reload_tsv: bool = True,
        ignore_modules: List = None,
    ) -> None:
        super().__init__(lazy)
        self._word_tokenizer = WordTokenizer()
        self._tokenizer = PretrainedTransformerTokenizer(
            "bert-base-uncased", do_lowercase=True
        )
        self._token_indexers = {
            "tokens": PretrainedTransformerIndexer(
                "bert-base-uncased", do_lowercase=True
            )
        }

        self._limit = 100 if tiny else -1

        self._image_feat_cache_dir = os.path.join(
            "cache", image_feat_path.split("/")[-1]
        )
        if len(cache_path) > 0:
            # self._image_feat_cache_dir = os.path.join(cache_path, 'cache', image_feat_path.split('/')[-1])
            self._image_feat_cache_dir = os.path.join(cache_path, "cache")
        self.img_data = None
        if reload_tsv:
            self.img_data = load_obj_tsv(
                image_feat_path,
                self._limit,
                save_cache=False,
                cache_path=self._image_feat_cache_dir,
            )
            self.img_data = {img["img_id"]: img for img in self.img_data}

        self._only_with_annotation = only_with_annotation
        self._only_with_box_annotation = only_with_box_annotation
        self._simple_exist_annotation = simple_exist_annotation
        self._max_boxes = max_boxes
        self._max_seq_length = max_seq_length

        self.annotations = {}
        self.attention_modes = {}
        with open(annotations_path, "r") as f:
            delim = ","
            if annotations_path[-3:] == "tsv":
                delim = "\t"
            reader = csv.DictReader(f, delimiter=delim)
            for annotation in reader:
                if annotation["annotation"]:
                    skip = False
                    for ignore in ignore_modules:
                        if "." + ignore in annotation["annotation"]:
                            skip = True
                    if skip:
                        continue
                    self.annotations[annotation["sentence"].lower()] = annotation[
                        "annotation"
                    ]
                    self.attention_modes[annotation["sentence"].lower()] = int(
                        annotation["all"]
                    )
        self.box_annotations = {}
        self.ignore = set()
        if box_annotations_path is not None and len(box_annotations_path) > 0:
            with open(box_annotations_path, "r") as f:
                delim = ","
                if box_annotations_path[-3:] == "tsv":
                    delim = "\t"
                reader = csv.DictReader(f, delimiter=delim)
                for annotation in reader:
                    if annotation["Incorrect program"] == "1":
                        self.ignore.add(annotation["identifier"])
                        continue
                    elif annotation["identifier"] in self.ignore:
                        continue
                    if annotation["identifier"] not in self.box_annotations:
                        self.box_annotations[annotation["identifier"]] = {}
                    module_no = int(
                        annotation["module_identifier"].split("m")[1].split("-")[0]
                    )
                    module_name = annotation["module"].split("[")[0]
                    if module_no not in self.box_annotations[annotation["identifier"]]:
                        self.box_annotations[annotation["identifier"]][module_no] = {}
                        self.box_annotations[annotation["identifier"]][module_no][
                            "module"
                        ] = module_name
                    image_no = int(annotation["module_identifier"][-1])
                    label = annotation["annotation"]
                    if module_name == "count":
                        if len(label) == 0:
                            self.box_annotations[annotation["identifier"]][module_no][
                                image_no
                            ] = 0
                        elif label.isnumeric():
                            self.box_annotations[annotation["identifier"]][module_no][
                                image_no
                            ] = int(label)
                        else:
                            self.box_annotations[annotation["identifier"]][module_no][
                                image_no
                            ] = annotation["annotation"].count("{")
                    elif len(annotation["annotation"]) > 0:
                        print(annotation["module_identifier"], annotation["annotation"])
                        self.box_annotations[annotation["identifier"]][module_no][
                            image_no
                        ] = json.loads(annotation["annotation"])
                    else:
                        self.box_annotations[annotation["identifier"]][module_no][
                            image_no
                        ] = []

    @overrides
    def _read(self, file_path: str):
        # # run this (and line in the end of the function) to output dataset for lxmert
        # os.makedirs('dataset/lxmert/', exist_ok=True)
        # lxmert_output_file = open('dataset/lxmert/' + file_path.split('/')[-1], 'wt')
        # instances = []

        count = 0
        with open(cached_path(file_path)) as f:
            examples = json.load(f)
            for ex in examples:
                if self.img_data is not None and (
                    ex["img0"] not in self.img_data or ex["img1"] not in self.img_data
                ):
                    continue
                logical_form = None
                attention_mode = None
                sent = ex["sent"].lower()
                if sent in self.annotations:
                    logical_form = self.annotations[sent]
                    if (
                        self._max_seq_length > -1
                        and len(logical_form.split("\n")) > self._max_seq_length
                    ):
                        continue
                    attention_mode = self.attention_modes[sent]
                else:
                    if self._only_with_annotation:
                        continue
                try:
                    if self._simple_exist_annotation:
                        find_index_start = logical_form.index("find[")
                        find_index_end = find_index_start + logical_form[
                            find_index_start:
                        ].index("]")
                        find_module = logical_form[
                            find_index_start : find_index_end + 1
                        ]
                        logical_form = "exist\n." + find_module
                    box_annotation = None
                    if (
                        ex["identifier"] in self.box_annotations
                        and ex["identifier"] not in self.ignore
                    ):
                        box_annotation = self.box_annotations[ex["identifier"]]
                    else:
                        if self._only_with_box_annotation:
                            continue
                    # print(sent, logical_form)
                    instance = self.text_to_instance(
                        sent,
                        ex["identifier"],
                        [ex["img0"], ex["img1"]],
                        logical_form,
                        attention_mode,
                        box_annotation,
                        ex["label"],
                    )
                except ParsingError as e:
                    print("ERROR PARSING:", sent)
                    continue
                if instance is not None:
                    # instances.append(ex)
                    count += 1
                    if count == self._limit:
                        break
                    yield instance

        # json.dump(instances, lxmert_output_file, indent=4)
        # print(len(instances), "instances saved")

    def text_to_instance(
        self,
        sentence: str,
        identifier: str,
        image_ids: List[str],
        logical_form: str = None,
        attention_mode: int = None,
        box_annotation: Dict = None,
        denotation: str = None,
    ) -> Instance:
        tokenized_sentence = self._tokenizer.tokenize(sentence)
        sentence_field = TextField(tokenized_sentence, self._token_indexers)

        world = VisualReasoningNlvr2Language(None, None, None, None, None, None)

        production_rule_fields: List[Field] = []
        instance_action_ids: Dict[str, int] = {}
        for production_rule in world.all_possible_productions():
            instance_action_ids[production_rule] = len(instance_action_ids)
            field = ProductionRuleField(production_rule, is_global_rule=True)
            production_rule_fields.append(field)

        action_field = ListField(production_rule_fields)

        boxes2 = []
        feats2 = []
        max_num_boxes = 0
        for key in image_ids:
            if self.img_data is not None:
                img_info = self.img_data[key]
            else:
                split_name = "train"
                if "dev" in key:
                    split_name = "valid"
                img_info = pickle.load(
                    open(
                        os.path.join(
                            self._image_feat_cache_dir, split_name + "_obj36.tsv", key
                        ),
                        "rb",
                    )
                )
            boxes = img_info["boxes"].copy()
            feats = img_info["features"].copy()
            assert len(boxes) == len(feats)

            # Normalize the boxes (to 0 ~ 1)
            img_h, img_w = img_info["img_h"], img_info["img_w"]
            boxes[..., (0, 2)] /= img_w
            boxes[..., (1, 3)] /= img_h
            np.testing.assert_array_less(boxes, 1 + 1e-5)
            np.testing.assert_array_less(-boxes, 0 + 1e-5)

            if boxes.shape[0] > self._max_boxes:
                boxes = boxes[: self._max_boxes, :]
                feats = feats[: self._max_boxes, :]
            max_num_boxes = max(max_num_boxes, boxes.shape[0])
            boxes2.append(boxes)
            feats2.append(feats)
        boxes3 = [
            np.zeros((max_num_boxes, img_boxes.shape[-1])) for img_boxes in boxes2
        ]
        feats3 = [
            np.zeros((max_num_boxes, img_feats.shape[-1])) for img_feats in feats2
        ]
        for i in range(len(boxes2)):
            boxes3[i][: boxes2[i].shape[0], :] = boxes2[i]
            feats3[i][: feats2[i].shape[0], :] = feats2[i]
        boxes2 = boxes3
        feats2 = feats3
        feats = np.stack(feats2)
        boxes = np.stack(boxes2)
        metadata: Dict[str, Any] = {
            "utterance": sentence,
            "tokenized_utterance": tokenized_sentence,
            "identifier": identifier,
        }

        fields: Dict[str, Field] = {
            "sentence": sentence_field,
            "actions": action_field,
            "metadata": MetadataField(metadata),
            "image_id": MetadataField(identifier[:-2]),
            "visual_feat": ArrayField(feats),
            "pos": ArrayField(boxes),
        }
        if denotation is not None:
            fields["denotation"] = LabelField(denotation, skip_indexing=True)

        if logical_form:
            lisp_exp = annotation_to_lisp_exp(logical_form)
            target_sequence = world.logical_form_to_action_sequence(lisp_exp)
            index_field = [
                IndexField(instance_action_ids[action], action_field)
                for action in target_sequence
            ]
            fields["target_action_sequence"] = ListField(index_field)

            module_attention = annotation_to_module_attention(logical_form)
            target_attention = target_sequence_to_target_attn(
                target_sequence, module_attention
            )
            gold_question_attentions = self._assign_attention_to_tokens(
                target_attention, sentence, attention_mode
            )
            attn_index_field = [
                ListField([IndexField(att, sentence_field) for att in target_att])
                for target_att in gold_question_attentions
            ]
            fields["gold_question_attentions"] = ListField(attn_index_field)
            if box_annotation is None and len(self.box_annotations) > 0:
                fields["gold_box_annotations"] = MetadataField([])
            elif box_annotation is not None:
                modules = logical_form.split("\n")
                children = [[] for _ in modules]
                for j, module in enumerate(modules):
                    num_periods = len(module) - len(module.strip("."))
                    for k in range(j + 1, len(modules)):
                        num_periods_k = len(modules[k]) - len(modules[k].strip("."))
                        if num_periods_k <= num_periods:
                            break
                        if num_periods_k == num_periods + 1:
                            children[j].append(k)
                for j in range(len(modules) - 1, -1, -1):
                    if modules[j].strip(".") == "in_left_image":
                        box_annotation[j] = {}
                        box_annotation[j]["module"] = modules[j].strip(".")
                        box_annotation[j][0] = box_annotation[j + 1][0]
                        box_annotation[j][1] = []
                        """for k in children[j]:
                            box_annotation[k][0] = box_annotation[k][0]
                            box_annotation[k][1] = []"""
                    elif modules[j].strip(".") == "in_right_image":
                        box_annotation[j] = {}
                        box_annotation[j]["module"] = modules[j].strip(".")
                        box_annotation[j][1] = box_annotation[j + 1][1]
                        box_annotation[j][0] = []
                    elif modules[j].strip(".") in {"in_one_image", "in_other_image"}:
                        box_annotation[j] = {}
                        box_annotation[j]["module"] = modules[j].strip(".")
                        box_annotation[j][0] = box_annotation[j + 1][0]
                        box_annotation[j][1] = box_annotation[j + 1][1]
                        """for k in children[j]:
                            box_annotation[k][0] = []
                            box_annotation[k][1] = box_annotation[k][1]"""
                keys = sorted(list(box_annotation.keys()))
                # print(identifier, keys)
                # print(box_annotation)
                # print(target_sequence)
                module_boxes = [
                    (
                        mod,
                        box_annotation[mod]["module"],
                        [box_annotation[mod][0], box_annotation[mod][1]],
                    )
                    for mod in keys
                ]
                gold_boxes, gold_counts = target_sequence_to_target_boxes(
                    target_sequence, module_boxes, children
                )
                # print(identifier, target_sequence, module_boxes, gold_boxes)
                fields["gold_box_annotations"] = MetadataField(gold_boxes)
            metadata["gold"] = world.action_sequence_to_logical_form(target_sequence)
            fields["valid_target_sequence"] = ArrayField(np.array(1, dtype=np.int32))
        else:
            fields["target_action_sequence"] = ListField([IndexField(0, action_field)])
            fields["gold_question_attentions"] = ListField(
                [ListField([IndexField(0, sentence_field)])]
            )
            fields["valid_target_sequence"] = ArrayField(np.array(0, dtype=np.int32))
            if len(self.box_annotations) > 0:
                fields["gold_box_annotations"] = MetadataField([])
        return Instance(fields)

    def _assign_attention_to_tokens(self, gold_attentions, sentence, attention_mode):
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

        assert alignment is not None

        for i in range(len(alignment)):
            alignment[i] = [int(j + 1) for j in alignment[i]]

        aligned_gold_attentions = []

        for i, atts in enumerate(gold_attentions):
            aligned_gold_attentions.append([])
            if attention_mode == 2:
                aligned_gold_attentions[i] = [1 + att for att in atts]
                continue
            for att in atts:
                if att >= 0:
                    aligned_gold_attentions[i] += alignment[att]
                else:
                    aligned_gold_attentions[i] += [att]

        return aligned_gold_attentions
