from copy import copy
from typing import NamedTuple, Set, List, Dict, Tuple, Any, Callable, Union

from allennlp.common.checks import ConfigurationError
from overrides import overrides
import inspect

import torch
import torch.nn.functional as F

import num2words

from allennlp.nn import util
from allennlp.semparse.domain_languages.domain_language import (
    DomainLanguage,
    predicate,
    predicate_with_side_args,
)

from lib.modules.counting import Counter
from lib.modules.seq2seq_encoders.lxmert_src.lxrt.modeling import GeLU, BertLayerNorm

INTEGER_MAP = {num2words.num2words(i): i for i in range(100)}


class Answer(torch.Tensor):
    pass


class ObjectSet(torch.Tensor):
    pass


class Number(NamedTuple):
    mean: torch.Tensor
    var: torch.Tensor


class VisualReasoningNlvr2Parameters(torch.nn.Module):
    def __init__(self, hidden_dim, initializer, max_boxes, dropout, nmn_settings):
        super().__init__()
        self.hidden_dim = hidden_dim
        if not nmn_settings["use_sum_counting"]:
            self.count_layer = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
            self.count_layer_2 = torch.nn.Linear(self.hidden_dim, 1)
            torch.nn.init.uniform_(self.count_layer_2.weight, -0.01, 0.01)
            self.count_norm = BertLayerNorm(self.hidden_dim)
            self.count_layer_2.apply(initializer)
        else:
            self.count_layer = torch.nn.Linear(self.hidden_dim, 1)
        torch.nn.init.uniform_(self.count_layer.weight, -0.01, 0.01)
        self.find_layer = torch.nn.Linear(self.hidden_dim * 2, 1)
        if "find_qst_query_num_layers" in nmn_settings:
            if nmn_settings["find_qst_query_num_layers"] == 1:
                self.find_layer_qst_query = torch.nn.Linear(
                    self.hidden_dim, self.hidden_dim
                )
            else:
                self.find_layer_qst_query = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(self.hidden_dim, self.hidden_dim)
                        for _ in range(nmn_settings["find_qst_query_num_layers"])
                    ]
                )
        else:
            self.find_layer_qst_query = torch.nn.Linear(
                self.hidden_dim, self.hidden_dim
            )
        self.find_layer_vis_query = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        # we want the resulting dot product to be close to zero, so this later
        # should be initialized with low numbers
        if hasattr(self.find_layer_qst_query, "weight"):
            torch.nn.init.uniform_(self.find_layer_qst_query.weight, -0.01, 0.01)
        else:
            for layer in self.find_layer_qst_query:
                torch.nn.init.uniform_(layer.weight, -0.01, 0.01)
        self.find_hidden_layer = torch.nn.Linear(
            self.hidden_dim * 2, self.hidden_dim * 2
        )
        self.gelu_layer = GeLU()
        torch.nn.init.uniform_(self.find_layer_vis_query.weight, -0.01, 0.01)
        self.filter_layer = torch.nn.Linear(self.hidden_dim * 2, 1)
        self.relate_layer = torch.nn.Linear(self.hidden_dim * 3, 1)
        self.relate_layer1 = torch.nn.Linear(self.hidden_dim * 3, self.hidden_dim)
        self.relate_layer2 = torch.nn.Linear(self.hidden_dim, 1)
        self.number_layer = torch.nn.Linear(self.hidden_dim, 1)
        if torch.cuda.is_available():
            self.project_embedding = torch.nn.Parameter(
                torch.zeros((self.hidden_dim)).cuda()
            )
        else:
            self.project_embedding = torch.nn.Parameter(torch.zeros((self.hidden_dim)))
        self.count_layer.apply(initializer)
        self.find_layer.apply(initializer)
        self.find_hidden_layer.bias.data.zero_()
        self.find_hidden_layer.weight.data.fill_diagonal_(1.0)
        self.filter_layer.apply(initializer)
        self.relate_layer.apply(initializer)
        self.number_layer.apply(initializer)
        self.epsilon = 1e-12
        self.num_objects_single_image = max_boxes
        if torch.cuda.is_available():
            self.integer_upper_bounds = (
                torch.arange(2 * self.num_objects_single_image + 1).float().cuda() + 0.5
            )
            self.integer_lower_bounds = (
                torch.arange(2 * self.num_objects_single_image + 1).float().cuda() - 0.5
            )
            self.left_only_constant = (
                torch.ones((2 * self.num_objects_single_image)).float().cuda()
            )
            self.right_only_constant = (
                torch.ones((2 * self.num_objects_single_image)).float().cuda()
            )
        else:
            self.integer_upper_bounds = (
                torch.arange(2 * self.num_objects_single_image + 1).float() + 0.5
            )
            self.integer_lower_bounds = (
                torch.arange(2 * self.num_objects_single_image + 1).float() - 0.5
            )
            self.left_only_constant = torch.ones(
                (2 * self.num_objects_single_image)
            ).float()
            self.right_only_constant = torch.ones(
                (2 * self.num_objects_single_image)
            ).float()
        self.left_only_constant[self.num_objects_single_image :] = 0.0
        self.right_only_constant[: self.num_objects_single_image] = 0.0
        self.counter = Counter(self.num_objects_single_image, already_sigmoided=True)
        self.dropout = torch.nn.Dropout(p=dropout)


class VisualReasoningNlvr2Language(DomainLanguage):
    def __init__(
        self,
        language_encoding: Union[torch.Tensor, Dict] = None,
        visual_encoding: Union[torch.Tensor, Dict] = None,
        cross_encoding: Union[torch.Tensor, Dict] = None,
        parameters: VisualReasoningNlvr2Parameters = None,
        tokens: List[str] = None,
        boxes: torch.Tensor = None,
        nmn_settings: Dict = None,
    ) -> None:
        fixed_variance = 0.25
        constants = {
            str(i): Number(mean=float(i), var=fixed_variance) for i in range(10)
        }
        super().__init__(start_types={Answer}, allowed_constants=constants)
        self.fixed_variance = fixed_variance
        self.one_image_setting = None
        self.other_image_setting = None
        if parameters is not None:
            self.language_encoding = language_encoding
            self.visual_encoding = visual_encoding
            self.cross_encoding = cross_encoding
            self.tokens = tokens
            self.parameters = parameters
            boxes_left = torch.nn.functional.pad(
                input=boxes[0],
                pad=(
                    0,
                    0,
                    0,
                    self.parameters.num_objects_single_image - boxes.shape[1],
                ),
                mode="constant",
                value=0,
            )
            boxes_right = torch.nn.functional.pad(
                input=boxes[1],
                pad=(
                    0,
                    0,
                    0,
                    self.parameters.num_objects_single_image - boxes.shape[1],
                ),
                mode="constant",
                value=0,
            )
            self.boxes = torch.cat((boxes_left, boxes_right), dim=0)
            self.left_only = False
            self.right_only = False
            self.call_from_project = False
            if isinstance(language_encoding, dict):
                self.device = 0
                if len(language_encoding) > 0:
                    self.device = list(language_encoding.values())[0].device
            else:
                self.device = language_encoding.device
            self.written_count = None
            self.computed_count = None
            self.exist_question = False
            self.nmn_settings: Dict = nmn_settings
            self.modules_debug_info = []

    def log(self, prob):
        return torch.log(prob + self.parameters.epsilon)

    @predicate
    def exist(self, objects: ObjectSet) -> Answer:
        self.exist_question = True
        return self.greater_equal(
            self.count(objects),
            Number(mean=1, var=torch.tensor(self.fixed_variance).to(self.device)),
        )

    @predicate_with_side_args(["question_attention"])
    def find(self, question_attention: torch.Tensor) -> ObjectSet:
        if self.nmn_settings["mask_non_attention"]:
            language_encoding = self.language_encoding[self.object_scores_index]
            visual_encoding = self.visual_encoding[self.object_scores_index]
            cross_encoding = self.cross_encoding[self.object_scores_index]
        else:
            language_encoding = self.language_encoding
            visual_encoding = self.visual_encoding
            cross_encoding = self.cross_encoding
        attended_question = self.get_attended_question(
            question_attention, language_encoding
        ).repeat_interleave(self.parameters.num_objects_single_image, dim=0)
        attended_objects = visual_encoding.view(-1, visual_encoding.shape[-1])

        if self.nmn_settings["mult_attended_objects"]:
            attended_objects *= attended_question

        if self.nmn_settings["find_dot_product"]:
            if self.nmn_settings["find_dot_activation"] == "relu":
                activation = F.relu
            elif self.nmn_settings["find_dot_activation"] == "tanh":
                activation = F.tanh
            else:
                raise ConfigurationError("find_dot_activation should be relu/tanh")

            if self.nmn_settings["find_dot_dropout"]:
                activation_func = lambda x: self.parameters.dropout(activation(x))
            else:
                activation_func = activation
            if hasattr(self.parameters.find_layer_qst_query, "weight"):
                attended_question = activation_func(
                    self.parameters.find_layer_qst_query(attended_question)
                )
            else:
                for layer in self.parameters.find_layer_qst_query:
                    attended_question = activation_func(layer(attended_question))

            if self.nmn_settings["find_dot_product_vis_query"]:
                attended_objects = activation(
                    self.parameters.find_layer_vis_query(attended_objects)
                )
            logits = (
                attended_question.unsqueeze(1)
                .matmul(attended_objects.unsqueeze(2))
                .view(2, -1)
            )
        elif self.nmn_settings["use_cross_encoding"]:
            find_layer_inputs = torch.cat(
                (
                    attended_objects,
                    cross_encoding.view(2, -1).repeat_interleave(
                        self.parameters.num_objects_single_image, dim=0
                    ),
                ),
                dim=-1,
            )
            logits = self.parameters.find_layer(find_layer_inputs).view(2, -1)
        else:
            find_layer_inputs = torch.cat((attended_objects, attended_question), dim=-1)
            logits = self.parameters.find_layer(find_layer_inputs).view(2, -1)

        if self.nmn_settings["use_sum_counting"]:
            sigmoided = torch.sigmoid(logits).view(-1)
        else:
            sigmoided = torch.sigmoid(logits).view(-1)

        if self.left_only:
            result = sigmoided * self.parameters.left_only_constant
        elif self.right_only:
            result = sigmoided * self.parameters.right_only_constant
        else:
            result = sigmoided

        if not self.call_from_project:
            exists = False
            for j in range(len(self.modules_debug_info)):
                if self.modules_debug_info[j]["index"] == self.object_scores_index:
                    assert (
                        self.left_only
                        or self.right_only
                        or self.one_image_setting is not None
                    )
                    exists = True
                    break
            if not exists:
                self.modules_debug_info.append(
                    {
                        "module": "find",
                        "index": self.object_scores_index,
                        "output": sigmoided,
                        "logits": logits,
                    }
                )
        return result

    @predicate_with_side_args(["question_attention"])
    def number(self, question_attention: torch.Tensor) -> Number:
        token = str(self.tokens[question_attention.view(-1).argmax()])
        if token.isnumeric():
            mean = int(token)
        else:
            mean = INTEGER_MAP[token]
        mean = torch.tensor(float(mean)).cuda()
        self.written_count = mean
        self.modules_debug_info.append({"module": "number", "output": mean})
        return Number(mean=mean, var=torch.tensor(self.fixed_variance).to(self.device))

    @predicate
    def count(self, objects: ObjectSet) -> Number:
        n_objs_img = self.parameters.num_objects_single_image
        if (
            self.nmn_settings["use_sum_counting"]
            and self.nmn_settings["use_count_module"]
        ):
            count_probs1, _ = self.parameters.counter(
                self.boxes[:n_objs_img].t().unsqueeze(0),
                objects[:n_objs_img].unsqueeze(0),
            )
            count_probs1 = count_probs1.squeeze()
            mean1 = (
                count_probs1
                * torch.arange(count_probs1.numel()).float().to(count_probs1.device)
            ).sum()
            count_probs2, _ = self.parameters.counter(
                self.boxes[n_objs_img:].t().unsqueeze(0),
                objects[n_objs_img:].unsqueeze(0),
            )
            count_probs2 = count_probs2.squeeze()
            mean2 = (
                count_probs2
                * torch.arange(count_probs2.numel()).float().to(count_probs2.device)
            ).sum()
            mean = mean1 + mean2
            self.computed_count = mean
            var = self.fixed_variance
            exists = False
            for j in range(len(self.modules_debug_info)):
                if self.modules_debug_info[j]["index"] == self.object_scores_index:
                    assert (
                        self.left_only
                        or self.right_only
                        or self.one_image_setting is not None
                    )
                    exists = True
                    if self.left_only:
                        self.modules_debug_info[j]["output"] = (
                            self.modules_debug_info[j]["output_right"] + mean1
                        )
                        self.modules_debug_info[j]["output_left"] = mean1
                    elif self.right_only:
                        self.modules_debug_info[j]["output"] = (
                            self.modules_debug_info[j]["output_left"] + mean2
                        )
                        self.modules_debug_info[j]["output_right"] = mean2
            if not exists:
                self.modules_debug_info.append(
                    {
                        "module": "count",
                        "index": self.object_scores_index,
                        "output": mean,
                        "output_left": mean1,
                        "output_right": mean2,
                    }
                )
            return Number(mean=mean, var=var)
        elif self.nmn_settings["use_sum_counting"]:
            mean1 = objects[:n_objs_img].sum()
            mean2 = objects[n_objs_img:].sum()
            var = self.fixed_variance
            mean = mean1 + mean2
            exists = False
            for j in range(len(self.modules_debug_info)):
                if self.modules_debug_info[j]["index"] == self.object_scores_index:
                    assert (
                        self.left_only
                        or self.right_only
                        or self.one_image_setting is not None
                    )
                    exists = True
                    if self.left_only:
                        self.modules_debug_info[j]["output"] = (
                            self.modules_debug_info[j]["output_right"] + mean1
                        )
                        self.modules_debug_info[j][
                            "output_left"
                        ] = self.modules_debug_info[j]["output_left"] = mean1
                    elif self.right_only:
                        self.modules_debug_info[j]["output"] = (
                            self.modules_debug_info[j]["output_left"] + mean1
                        )
                        self.modules_debug_info[j][
                            "output_right"
                        ] = self.modules_debug_info[j]["output_left"] = mean1
            if not exists:
                self.modules_debug_info.append(
                    {
                        "module": "count",
                        "index": self.object_scores_index,
                        "output": mean,
                        "output_left": mean1,
                        "output_right": mean2,
                    }
                )
            return Number(mean=mean, var=var)
        n_objects = objects.size(0)
        if self.nmn_settings["mask_non_attention"]:
            visual_encoding = self.visual_encoding[self.object_scores_index]
        else:
            visual_encoding = self.visual_encoding
        vis_features = visual_encoding.view(n_objects, -1)
        attended_vis_left = util.weighted_sum(
            vis_features[:n_objs_img, :], objects[:n_objs_img]
        )
        attended_vis_right = util.weighted_sum(
            vis_features[n_objs_img:, :], objects[n_objs_img:]
        )
        gelu_layer = GeLU()
        hid_left = gelu_layer(self.parameters.count_layer(attended_vis_left))
        hid_right = gelu_layer(self.parameters.count_layer(attended_vis_right))
        hid_left = self.parameters.count_norm(self.parameters.dropout(hid_left))
        hid_right = self.parameters.count_norm(self.parameters.dropout(hid_right))
        number_left = F.relu(self.parameters.count_layer_2(hid_left))
        number_right = F.relu(self.parameters.count_layer_2(hid_right))
        number = number_left + number_right + 1e-10
        exists = False
        for j in range(len(self.modules_debug_info)):
            if self.modules_debug_info[j]["index"] == self.object_scores_index:
                assert (
                    self.left_only
                    or self.right_only
                    or self.one_image_setting is not None
                )
                exists = True
        if not exists:
            self.modules_debug_info.append(
                {
                    "module": "count",
                    "index": self.object_scores_index,
                    "output": number,
                    "output_left": number_left,
                    "output_right": number_right,
                }
            )
        # print('mean', number)
        return Number(
            mean=number, var=torch.tensor(self.fixed_variance).to(self.device)
        )

    @predicate
    def sum(self, a: Number, b: Number) -> Number:
        return Number(mean=a.mean + b.mean, var=a.var + b.var)

    @predicate
    def difference(self, a: Number, b: Number) -> Number:
        return Number(mean=a.mean - b.mean, var=a.var + b.var)

    @predicate
    def division(self, a: Number, b: Number) -> Number:
        # See second page of http://www.stat.cmu.edu/~hseltman/files/ratio.pdf
        return Number(
            mean=a.mean / b.mean + b.var * a.mean / (b.mean ** 3),
            var=(a.mean ** 2 / b.mean ** 2)
            * (a.var / a.mean ** 2 + b.var / b.mean ** 2),
        )

    @predicate
    def greater_equal(self, a: Number, b: Number) -> Answer:
        return self.greater(a, b) + self.equal(a, b)

    @predicate
    def less_equal(self, a: Number, b: Number) -> Answer:
        return self.less(a, b) + self.equal(a, b)

    @predicate
    def equal(self, a: Number, b: Number) -> Answer:
        a_distr = torch.distributions.normal.Normal(a.mean, a.var)
        b_distr = torch.distributions.normal.Normal(b.mean, b.var)
        probs_a = a_distr.cdf(self.parameters.integer_upper_bounds) - a_distr.cdf(
            self.parameters.integer_lower_bounds
        )
        probs_b = b_distr.cdf(self.parameters.integer_upper_bounds) - b_distr.cdf(
            self.parameters.integer_lower_bounds
        )
        probs_a[0] = a_distr.cdf(self.parameters.integer_upper_bounds[0])
        probs_a[-1] = 1.0 - a_distr.cdf(self.parameters.integer_lower_bounds[-1])
        probs_b[0] = b_distr.cdf(self.parameters.integer_upper_bounds[0])
        probs_b[-1] = 1.0 - b_distr.cdf(self.parameters.integer_lower_bounds[-1])
        result = (probs_a * probs_b).sum()
        return result

    @predicate
    def less(self, a: Number, b: Number) -> Answer:
        a_distr = torch.distributions.normal.Normal(a.mean, a.var)
        b_distr = torch.distributions.normal.Normal(b.mean, b.var)
        probs_a = a_distr.cdf(self.parameters.integer_upper_bounds) - a_distr.cdf(
            self.parameters.integer_lower_bounds
        )
        probs_b = b_distr.cdf(self.parameters.integer_upper_bounds) - b_distr.cdf(
            self.parameters.integer_lower_bounds
        )
        probs_a[0] = a_distr.cdf(self.parameters.integer_upper_bounds[0])
        probs_a[-1] = 1.0 - a_distr.cdf(self.parameters.integer_lower_bounds[-1])
        probs_b[0] = b_distr.cdf(self.parameters.integer_upper_bounds[0])
        probs_b[-1] = 1.0 - b_distr.cdf(self.parameters.integer_lower_bounds[-1])
        result = 0.0
        for i in range(len(self.parameters.integer_upper_bounds) - 1):
            result += probs_a[i] * probs_b[i + 1 :].sum()
        return result

    @predicate
    def greater(self, a: Number, b: Number) -> Answer:
        a_distr = torch.distributions.normal.Normal(a.mean, a.var)
        b_distr = torch.distributions.normal.Normal(b.mean, b.var)
        probs_a = a_distr.cdf(self.parameters.integer_upper_bounds) - a_distr.cdf(
            self.parameters.integer_lower_bounds
        )
        probs_b = b_distr.cdf(self.parameters.integer_upper_bounds) - b_distr.cdf(
            self.parameters.integer_lower_bounds
        )
        probs_a[0] = a_distr.cdf(self.parameters.integer_upper_bounds[0])
        probs_a[-1] = 1.0 - a_distr.cdf(self.parameters.integer_lower_bounds[-1])
        probs_b[0] = b_distr.cdf(self.parameters.integer_upper_bounds[0])
        probs_b[-1] = 1.0 - b_distr.cdf(self.parameters.integer_lower_bounds[-1])
        result = 0.0
        for i in range(1, len(self.parameters.integer_upper_bounds)):
            result += probs_a[i] * probs_b[:i].sum()
        return result

    @predicate_with_side_args(["question_attention"])
    def project(self, b: ObjectSet, question_attention: torch.Tensor) -> ObjectSet:
        self.call_from_project = True
        a = self.find(question_attention=question_attention)
        self.call_from_project = False
        if self.nmn_settings["mask_non_attention"]:
            language_encoding = self.language_encoding[self.object_scores_index]
            visual_encoding = self.visual_encoding[self.object_scores_index]
            cross_encoding = self.cross_encoding[self.object_scores_index]
        else:
            language_encoding = self.language_encoding
            visual_encoding = self.visual_encoding
            cross_encoding = self.cross_encoding
        if self.nmn_settings["simple_with_relation"]:
            b_repeat = torch.cat(
                (
                    (b * self.parameters.left_only_constant)
                    .view(1, -1)
                    .repeat(b.numel() // 2, 1),
                    (b * self.parameters.right_only_constant)
                    .view(1, -1)
                    .repeat(b.numel() // 2, 1),
                ),
                dim=0,
            )
            b_all_but_one = (1.0 - torch.eye(b.numel()).to(b.device)) * b_repeat
            return a * b_all_but_one.max(1)[0].view(-1)
        visual_encoding_rows = visual_encoding.repeat(
            1, 1, visual_encoding.shape[1]
        ).view(-1, visual_encoding.shape[-1])
        visual_encoding_columns = visual_encoding.repeat(
            1, visual_encoding.shape[1], 1
        ).view(-1, visual_encoding.shape[-1])
        b_repeat = torch.cat(
            (
                (b * self.parameters.left_only_constant)
                .view(1, -1)
                .repeat(b.numel() // 2, 1),
                (b * self.parameters.right_only_constant)
                .view(1, -1)
                .repeat(b.numel() // 2, 1),
            ),
            dim=0,
        )
        b_all_but_one = (1.0 - torch.eye(b.numel()).to(b.device)) * b_repeat
        best_b_obj = b_all_but_one.argmax(1)
        b_encodings = visual_encoding.view(-1, visual_encoding.shape[-1])[best_b_obj, :]
        relate_layer_inputs = torch.cat(
            (
                visual_encoding.view(-1, visual_encoding.shape[-1]),
                b_encodings,
                self.parameters.project_embedding.unsqueeze(0).repeat(b.numel(), 1),
            ),
            dim=-1,
        )
        gelu_layer = GeLU()
        object_scores = (
            torch.sigmoid(
                self.parameters.relate_layer2(
                    gelu_layer(self.parameters.relate_layer1(relate_layer_inputs))
                )
            ).view(-1)
            * a
            * b_all_but_one.max(1)[0].view(-1)
        )
        exists = False
        for j in range(len(self.modules_debug_info)):
            if self.modules_debug_info[j]["index"] == self.object_scores_index:
                assert (
                    self.left_only
                    or self.right_only
                    or self.one_image_setting is not None
                )
                if self.left_only:
                    self.modules_debug_info[j]["output"] = torch.cat(
                        (
                            object_scores[: object_scores.numel() // 2],
                            self.modules_debug_info[j]["output"][
                                object_scores.numel() // 2 :
                            ],
                        ),
                        dim=0,
                    )
                elif self.right_only:
                    self.modules_debug_info[j]["output"] = torch.cat(
                        (
                            self.modules_debug_info[j]["output"][
                                : object_scores.numel() // 2
                            ],
                            object_scores[object_scores.numel() // 2 :],
                        ),
                        dim=0,
                    )
                exists = True
        if not exists:
            self.modules_debug_info.append(
                {
                    "module": "project",
                    "index": self.object_scores_index,
                    "output": object_scores,
                }
            )
        return object_scores

    @predicate_with_side_args(["question_attention"])
    def with_relation(
        self, a: ObjectSet, b: ObjectSet, question_attention: torch.Tensor
    ) -> ObjectSet:
        # commenting out until we go over this module
        if self.nmn_settings["mask_non_attention"]:
            language_encoding = self.language_encoding[self.object_scores_index]
            visual_encoding = self.visual_encoding[self.object_scores_index]
            cross_encoding = self.cross_encoding[self.object_scores_index]
        else:
            language_encoding = self.language_encoding
            visual_encoding = self.visual_encoding
            cross_encoding = self.cross_encoding
        if self.nmn_settings["simple_with_relation"]:
            b_repeat = torch.cat(
                (
                    (b * self.parameters.left_only_constant)
                    .view(1, -1)
                    .repeat(b.numel() // 2, 1),
                    (b * self.parameters.right_only_constant)
                    .view(1, -1)
                    .repeat(b.numel() // 2, 1),
                ),
                dim=0,
            )
            b_all_but_one = (1.0 - torch.eye(b.numel()).to(b.device)) * b_repeat
            return a * b_all_but_one.max(1)[0].view(-1)
        attended_question = self.get_attended_question(
            question_attention, language_encoding
        ).repeat_interleave(self.parameters.num_objects_single_image, dim=0)
        visual_encoding_rows = visual_encoding.repeat(
            1, 1, visual_encoding.shape[1]
        ).view(-1, visual_encoding.shape[-1])
        visual_encoding_columns = visual_encoding.repeat(
            1, visual_encoding.shape[1], 1
        ).view(-1, visual_encoding.shape[-1])
        b_repeat = torch.cat(
            (
                (b * self.parameters.left_only_constant)
                .view(1, -1)
                .repeat(b.numel() // 2, 1),
                (b * self.parameters.right_only_constant)
                .view(1, -1)
                .repeat(b.numel() // 2, 1),
            ),
            dim=0,
        )
        b_all_but_one = (1.0 - torch.eye(b.numel()).to(b.device)) * b_repeat
        best_b_obj = b_all_but_one.argmax(1)
        b_encodings = visual_encoding.view(-1, visual_encoding.shape[-1])[best_b_obj, :]
        if self.nmn_settings["use_cross_encoding"]:
            relate_layer_inputs = torch.cat(
                (
                    visual_encoding.view(-1, visual_encoding.shape[-1]),
                    b_encodings,
                    cross_encoding.view(2, -1)
                    .unsqueeze(1)
                    .repeat(1, visual_encoding.shape[1], 1)
                    .view(-1, cross_encoding.shape[-1]),
                ),
                dim=-1,
            )
        else:
            relate_layer_inputs = torch.cat(
                (
                    visual_encoding.view(-1, visual_encoding.shape[-1]),
                    b_encodings,
                    attended_question,
                ),
                dim=-1,
            )
        gelu_layer = GeLU()
        object_scores = (
            torch.sigmoid(
                self.parameters.relate_layer2(
                    gelu_layer(self.parameters.relate_layer1(relate_layer_inputs))
                )
            ).view(-1)
            * a
            * b_all_but_one.max(1)[0].view(-1)
        )
        exists = False
        for j in range(len(self.modules_debug_info)):
            if self.modules_debug_info[j]["index"] == self.object_scores_index:
                assert (
                    self.left_only
                    or self.right_only
                    or self.one_image_setting is not None
                )
                if self.left_only:
                    self.modules_debug_info[j]["output"] = torch.cat(
                        (
                            object_scores[: object_scores.numel() // 2],
                            self.modules_debug_info[j]["output"][
                                object_scores.numel() // 2 :
                            ],
                        ),
                        dim=0,
                    )
                elif self.right_only:
                    self.modules_debug_info[j]["output"] = torch.cat(
                        (
                            self.modules_debug_info[j]["output"][
                                : object_scores.numel() // 2
                            ],
                            object_scores[object_scores.numel() // 2 :],
                        ),
                        dim=0,
                    )
                exists = True
        if not exists:
            self.modules_debug_info.append(
                {
                    "module": "with_relation",
                    "index": self.object_scores_index,
                    "output": object_scores,
                }
            )
        return object_scores

    @predicate_with_side_args(["question_attention"])
    def filter(self, objects: ObjectSet, question_attention: torch.Tensor) -> ObjectSet:
        n_objects = objects.size(0)
        if self.nmn_settings["mask_non_attention"]:
            language_encoding = self.language_encoding[self.object_scores_index]
            visual_encoding = self.visual_encoding[self.object_scores_index]
            cross_encoding = self.cross_encoding[self.object_scores_index]
        else:
            language_encoding = self.language_encoding
            visual_encoding = self.visual_encoding
            cross_encoding = self.cross_encoding
        if self.nmn_settings["filter_find_same_params"]:
            attended_question = self.get_attended_question(
                question_attention, language_encoding
            ).repeat_interleave(self.parameters.num_objects_single_image, dim=0)
            attended_objects = visual_encoding.view(-1, visual_encoding.shape[-1])
            if self.nmn_settings["find_dot_product"]:
                if self.nmn_settings["find_dot_activation"] == "relu":
                    activation = F.relu
                elif self.nmn_settings["find_dot_activation"] == "tanh":
                    activation = F.tanh
                else:
                    raise ConfigurationError("find_dot_activation should be relu/tanh")

                if self.nmn_settings["find_dot_dropout"]:
                    activation = lambda x: self.parameters.dropout(activation(x))
                attended_question = activation(
                    self.parameters.find_layer_qst_query(attended_question)
                )

                if self.nmn_settings["find_dot_product_vis_query"]:
                    attended_objects = F.relu(
                        self.parameters.find_layer_vis_query(attended_objects)
                    )
                logits = (
                    attended_question.unsqueeze(1)
                    .matmul(attended_objects.unsqueeze(2))
                    .view(2, -1)
                )
            elif self.nmn_settings["use_cross_encoding"]:
                find_layer_inputs = torch.cat(
                    (
                        attended_objects,
                        cross_encoding.view(2, -1).repeat_interleave(
                            self.parameters.num_objects_single_image, dim=0
                        ),
                    ),
                    dim=-1,
                )
                logits = self.parameters.find_layer(find_layer_inputs).view(2, -1)
            else:
                find_layer_inputs = torch.cat(
                    (attended_objects, attended_question), dim=-1
                )
                logits = self.parameters.find_layer(find_layer_inputs).view(2, -1)
            filter_probs = torch.sigmoid(logits).view(-1)
        elif self.nmn_settings["use_sum_counting"]:
            attended_question = self.get_attended_question(
                question_attention, language_encoding
            )
            if self.nmn_settings["mult_attended_objects"]:
                attended_objects = visual_encoding * attended_question.unsqueeze(1)
            else:
                attended_objects = visual_encoding
            logits = self.parameters.filter_layer(
                torch.cat(
                    (
                        visual_encoding.view(-1, visual_encoding.shape[-1]),
                        attended_question.repeat_interleave(
                            visual_encoding.shape[1], dim=0
                        ),
                    ),
                    dim=-1,
                )
            )
            filter_probs = torch.sigmoid(logits).view(-1)
        left_filter_probs = torch.nn.functional.pad(
            input=filter_probs[: visual_encoding.shape[1]],
            pad=(
                0,
                self.parameters.num_objects_single_image
                - filter_probs[: visual_encoding.shape[1]].numel(),
            ),
            mode="constant",
            value=0,
        )
        right_filter_probs = torch.nn.functional.pad(
            input=filter_probs[visual_encoding.shape[1] :],
            pad=(
                0,
                self.parameters.num_objects_single_image
                - filter_probs[visual_encoding.shape[1] :].numel(),
            ),
            mode="constant",
            value=0,
        )
        filter_probs = torch.cat((left_filter_probs, right_filter_probs), dim=0)
        result = filter_probs.view(n_objects) * objects
        exists = False
        for j in range(len(self.modules_debug_info)):
            if self.modules_debug_info[j]["index"] == self.object_scores_index:
                assert (
                    self.left_only
                    or self.right_only
                    or self.one_image_setting is not None
                )
                if self.left_only:
                    self.modules_debug_info[j]["output"] = torch.cat(
                        (
                            result[: result.numel() // 2],
                            self.modules_debug_info[j]["output"][result.numel() // 2 :],
                        ),
                        dim=0,
                    )
                elif self.right_only:
                    self.modules_debug_info[j]["output"] = torch.cat(
                        (
                            self.modules_debug_info[j]["output"][: result.numel() // 2],
                            result[result.numel() // 2 :],
                        ),
                        dim=0,
                    )
                exists = True
        if not exists:
            self.modules_debug_info.append(
                {
                    "module": "filter",
                    "index": self.object_scores_index,
                    "filter_probs": filter_probs,
                    "output": result,
                    "logits": logits,
                }
            )
        return result

    def get_attended_question(self, question_attention, language_encoding=None):
        if language_encoding is None:
            language_encoding = self.language_encoding
        attended_question_left = util.weighted_sum(
            language_encoding[0, :, :], question_attention
        )
        attended_question_right = util.weighted_sum(
            language_encoding[1, :, :], question_attention
        )
        attended_question = torch.stack(
            (attended_question_left, attended_question_right), dim=0
        )
        return attended_question

    @predicate
    def bool_and(self, a: Answer, b: Answer) -> Answer:
        return a * b

    @predicate
    def bool_or(self, a: Answer, b: Answer) -> Answer:
        return a + b - a * b

    @predicate
    def in_left_image(self, objects: ObjectSet) -> ObjectSet:
        return objects * self.parameters.left_only_constant

    @predicate
    def in_right_image(self, objects: ObjectSet) -> ObjectSet:
        return objects * self.parameters.right_only_constant

    @predicate
    def in_at_least_one_image(self, b: Answer) -> Answer:
        return b

    @predicate
    def in_each_image(self, b: Answer) -> Answer:
        return b

    @predicate
    def in_one_image(self, objects: ObjectSet) -> ObjectSet:
        return objects

    @predicate
    def in_other_image(self, objects: ObjectSet) -> ObjectSet:
        return objects

    @predicate
    def in_one_other_image(self, b: Answer) -> Answer:
        return b

    @predicate
    def intersect(self, objects1: ObjectSet, objects2: ObjectSet) -> ObjectSet:
        return objects1 * objects2

    @predicate
    def discard(self, objects1: ObjectSet, objects2: ObjectSet) -> ObjectSet:
        return torch.cat(
            ((objects1 - objects2).unsqueeze(1), 0 * objects1.unsqueeze(1)), dim=1
        ).max(1)[0]

    @overrides
    def _execute_sequence(
        self, action_sequence: List[str], side_arguments: List[Dict], full_seq_length=-1
    ) -> Tuple[Any, List[str], List[Dict]]:
        """
        This does the bulk of the work of :func:`execute_action_sequence`, recursively executing
        the functions it finds and trimming actions off of the action sequence.  The return value
        is a tuple of (execution, remaining_actions), where the second value is necessary to handle
        the recursion.
        """
        first_action = action_sequence[0]
        if full_seq_length == -1:
            full_seq_length = len(action_sequence)
        remaining_actions = action_sequence[1:]
        remaining_side_args = side_arguments[1:] if side_arguments else None
        right_side = first_action.split(" -> ")[1]
        if right_side in self._functions:
            function = self._functions[right_side]
            # mypy doesn't like this check, saying that Callable isn't a reasonable thing to pass
            # here.  But it works just fine; I'm not sure why mypy complains about it.
            if isinstance(function, Callable):  # type: ignore
                function_arguments = inspect.signature(function).parameters
                if not function_arguments:
                    # This was a zero-argument function / constant that was registered as a lambda
                    # function, for consistency of execution in `execute()`.
                    execution_value = function()
                elif side_arguments:
                    kwargs = {}
                    non_kwargs = []
                    for argument_name in function_arguments:
                        if argument_name in side_arguments[0]:
                            kwargs[argument_name] = side_arguments[0][argument_name]
                        else:
                            non_kwargs.append(argument_name)
                    if kwargs and non_kwargs:
                        # This is a function that has both side arguments and logical form
                        # arguments - we curry the function so only the logical form arguments are
                        # left.
                        def curried_function(*args):
                            self.object_scores_index = (
                                1 + full_seq_length - len(action_sequence)
                            )
                            return function(*args, **kwargs)

                        execution_value = curried_function
                    elif kwargs:
                        # This is a function that _only_ has side arguments - we just call the
                        # function and return a value.
                        self.object_scores_index = (
                            1 + full_seq_length - len(action_sequence)
                        )
                        execution_value = function(**kwargs)
                    else:
                        # This is a function that has logical form arguments, but no side arguments
                        # that match what we were given - just return the function itself.
                        execution_value = function
                else:
                    execution_value = function
            self.object_scores_index = 1 + full_seq_length - len(action_sequence)
            return execution_value, remaining_actions, remaining_side_args
        else:
            # This is a non-terminal expansion, like 'int -> [<int:int>, int, int]'.  We need to
            # get the function and its arguments, then call the function with its arguments.
            # Because we linearize the abstract syntax tree depth first, left-to-right, we can just
            # recursively call `_execute_sequence` for the function and all of its arguments, and
            # things will just work.
            right_side_parts = right_side.split(", ")

            # We don't really need to know what the types are, just how many of them there are, so
            # we recurse the right number of times.
            function, remaining_actions, remaining_side_args = self._execute_sequence(
                remaining_actions, remaining_side_args, full_seq_length
            )
            index = self.object_scores_index
            if function.__name__ == "in_one_other_image":
                # Execute subtree twice; once with `one` image as left image, `other` image as right image; and
                # again with `one` image as right image, `other` image as left image
                original_remaining_actions = copy(remaining_actions)
                original_remaining_side_args = copy(remaining_side_args)
                self.one_image_setting = "left"
                self.other_image_setting = "right"
                arguments = []
                for _ in right_side_parts[1:]:
                    (
                        argument,
                        remaining_actions,
                        remaining_side_args,
                    ) = self._execute_sequence(
                        remaining_actions, remaining_side_args, full_seq_length
                    )
                    arguments.append(argument)
                self.object_scores_index = index
                res1 = function(*arguments)
                self.one_image_setting = "right"
                self.other_image_setting = "left"
                arguments = []
                for _ in right_side_parts[1:]:
                    (
                        argument,
                        remaining_actions,
                        remaining_side_args,
                    ) = self._execute_sequence(
                        original_remaining_actions,
                        original_remaining_side_args,
                        full_seq_length,
                    )
                    arguments.append(argument)
                self.object_scores_index = index
                res2 = function(*arguments)
                self.one_image_setting = None
                self.other_image_setting = None
                self.object_scores_index = 1 + full_seq_length - len(action_sequence)
                return self.bool_or(res1, res2), remaining_actions, remaining_side_args
            if function.__name__ == "in_one_image":
                original_remaining_actions = copy(remaining_actions)
                original_remaining_side_args = copy(remaining_side_args)
                if self.one_image_setting == "left":
                    self.left_only = True
                    self.right_only = False
                    arguments = []
                    for _ in right_side_parts[1:]:
                        (
                            argument,
                            remaining_actions,
                            remaining_side_args,
                        ) = self._execute_sequence(
                            remaining_actions, remaining_side_args, full_seq_length
                        )
                        arguments.append(argument)
                    self.left_only = False
                    self.object_scores_index = index
                    res = function(*arguments)
                    self.object_scores_index = (
                        1 + full_seq_length - len(action_sequence)
                    )
                    return res, remaining_actions, remaining_side_args
                elif self.one_image_setting == "right":
                    self.left_only = False
                    self.right_only = True
                    arguments = []
                    for _ in right_side_parts[1:]:
                        (
                            argument,
                            remaining_actions,
                            remaining_side_args,
                        ) = self._execute_sequence(
                            remaining_actions, remaining_side_args, full_seq_length
                        )
                        arguments.append(argument)
                    self.right_only = False
                    self.object_scores_index = index
                    res = function(*arguments)
                    self.object_scores_index = (
                        1 + full_seq_length - len(action_sequence)
                    )
                    return res, remaining_actions, remaining_side_args
                else:
                    assert self.one_image_setting is None
                    # if one_image_setting is None, treat basically as in_at_least_one_image
                    self.left_only = True
                    arguments = []
                    for _ in right_side_parts[1:]:
                        (
                            argument,
                            remaining_actions,
                            remaining_side_args,
                        ) = self._execute_sequence(
                            remaining_actions, remaining_side_args, full_seq_length
                        )
                        arguments.append(argument)
                    self.object_scores_index = index
                    res1 = function(*arguments)
                    self.left_only = False
                    self.right_only = True
                    arguments = []
                    for _ in right_side_parts[1:]:
                        (
                            argument,
                            remaining_actions,
                            remaining_side_args,
                        ) = self._execute_sequence(
                            original_remaining_actions,
                            original_remaining_side_args,
                            full_seq_length,
                        )
                        arguments.append(argument)
                    self.object_scores_index = index
                    res2 = function(*arguments)
                    self.right_only = False
                    self.object_scores_index = (
                        1 + full_seq_length - len(action_sequence)
                    )
                    return (
                        self.bool_or(res1, res2),
                        remaining_actions,
                        remaining_side_args,
                    )
            if function.__name__ == "in_other_image":
                original_remaining_actions = copy(remaining_actions)
                original_remaining_side_args = copy(remaining_side_args)
                if self.other_image_setting == "left":
                    self.left_only = True
                    self.right_only = False
                    arguments = []
                    for _ in right_side_parts[1:]:
                        (
                            argument,
                            remaining_actions,
                            remaining_side_args,
                        ) = self._execute_sequence(
                            remaining_actions, remaining_side_args, full_seq_length
                        )
                        arguments.append(argument)
                    self.left_only = False
                    self.object_scores_index = index
                    res = function(*arguments)
                    self.object_scores_index = (
                        1 + full_seq_length - len(action_sequence)
                    )
                    return res, remaining_actions, remaining_side_args
                elif self.other_image_setting == "right":
                    self.left_only = False
                    self.right_only = True
                    arguments = []
                    for _ in right_side_parts[1:]:
                        (
                            argument,
                            remaining_actions,
                            remaining_side_args,
                        ) = self._execute_sequence(
                            remaining_actions, remaining_side_args, full_seq_length
                        )
                        arguments.append(argument)
                    self.right_only = False
                    self.object_scores_index = index
                    res = function(*arguments)
                    self.object_scores_index = (
                        1 + full_seq_length - len(action_sequence)
                    )
                    return res, remaining_actions, remaining_side_args
                else:
                    assert self.other_image_setting is None
                    arguments = []
                    for _ in right_side_parts[1:]:
                        (
                            argument,
                            remaining_actions,
                            remaining_side_args,
                        ) = self._execute_sequence(
                            remaining_actions, remaining_side_args, full_seq_length
                        )
                        arguments.append(argument)
                    self.object_scores_index = index
                    res = function(*arguments)
                    self.object_scores_index = (
                        1 + full_seq_length - len(action_sequence)
                    )
                    return res, remaining_actions, remaining_side_args
            if function.__name__ == "in_at_least_one_image":
                # Execute subtree separately for each image and combine the results with `or`
                original_remaining_actions = copy(remaining_actions)
                original_remaining_side_args = copy(remaining_side_args)
                self.left_only = True
                arguments = []
                for _ in right_side_parts[1:]:
                    (
                        argument,
                        remaining_actions,
                        remaining_side_args,
                    ) = self._execute_sequence(
                        remaining_actions, remaining_side_args, full_seq_length
                    )
                    arguments.append(argument)
                self.object_scores_index = index
                res1 = function(*arguments)
                self.left_only = False
                self.right_only = True
                arguments = []
                for _ in right_side_parts[1:]:
                    (
                        argument,
                        remaining_actions,
                        remaining_side_args,
                    ) = self._execute_sequence(
                        original_remaining_actions,
                        original_remaining_side_args,
                        full_seq_length,
                    )
                    arguments.append(argument)
                self.object_scores_index = index
                res2 = function(*arguments)
                self.right_only = False
                self.object_scores_index = 1 + full_seq_length - len(action_sequence)
                return self.bool_or(res1, res2), remaining_actions, remaining_side_args
            elif function.__name__ == "in_each_image":
                # Execute subtree separately for each image and combine the results with `and`
                original_remaining_actions = copy(remaining_actions)
                original_remaining_side_args = copy(remaining_side_args)
                self.left_only = True
                arguments = []
                for _ in right_side_parts[1:]:
                    (
                        argument,
                        remaining_actions,
                        remaining_side_args,
                    ) = self._execute_sequence(
                        remaining_actions, remaining_side_args, full_seq_length
                    )
                    arguments.append(argument)
                self.object_scores_index = index
                res1 = function(*arguments)
                self.left_only = False
                self.right_only = True
                arguments = []
                for _ in right_side_parts[1:]:
                    (
                        argument,
                        remaining_actions,
                        remaining_side_args,
                    ) = self._execute_sequence(
                        original_remaining_actions,
                        original_remaining_side_args,
                        full_seq_length,
                    )
                    arguments.append(argument)
                self.object_scores_index = index
                res2 = function(*arguments)
                self.right_only = False
                self.object_scores_index = 1 + full_seq_length - len(action_sequence)
                return self.bool_and(res1, res2), remaining_actions, remaining_side_args
            arguments = []
            for _ in right_side_parts[1:]:
                (
                    argument,
                    remaining_actions,
                    remaining_side_args,
                ) = self._execute_sequence(
                    remaining_actions, remaining_side_args, full_seq_length
                )
                arguments.append(argument)
            self.object_scores_index = index
            res = function(*arguments)
            self.object_scores_index = 1 + full_seq_length - len(action_sequence)
            return res, remaining_actions, remaining_side_args
