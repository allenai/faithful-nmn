from typing import NamedTuple, Set, List, Dict, Tuple, Any, Callable
from overrides import overrides
import inspect

import torch
import torch.nn.functional as F

from allennlp.nn import util
from allennlp.semparse.domain_languages.domain_language import (
    DomainLanguage,
    predicate,
    predicate_with_side_args,
)


class Answer(NamedTuple):
    pass


class ObjectSet(torch.Tensor):
    pass


class Number(NamedTuple):
    mean: torch.Tensor
    var: torch.Tensor


class VisualReasoningGqaParameters(torch.nn.Module):
    def __init__(self, hidden_dim, initializer):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.count_layer = torch.nn.Linear(self.hidden_dim, 1)
        self.find_layer = torch.nn.Linear(self.hidden_dim * 2, 1)
        self.filter_layer = torch.nn.Linear(self.hidden_dim * 2, 1)
        self.relate_layer = torch.nn.Linear(self.hidden_dim * 3, 1)
        self.number_layer = torch.nn.Linear(self.hidden_dim, 1)
        self.count_layer.apply(initializer)
        self.find_layer.apply(initializer)
        self.filter_layer.apply(initializer)
        self.relate_layer.apply(initializer)
        self.number_layer.apply(initializer)
        self.epsilon = 1e-6
        self.fixed_variance = 0.01
        self.num_objects_single_image = 36
        self.integer_upper_bounds = (
            torch.arange(self.num_objects_single_image + 1).float().cuda() + 0.5
        )
        self.integer_lower_bounds = (
            torch.arange(self.num_objects_single_image + 1).float().cuda() - 0.5
        )
        self.left_only_constant = (
            torch.ones((self.num_objects_single_image)).float().cuda()
        )
        self.right_only_constant = (
            torch.ones((self.num_objects_single_image)).float().cuda()
        )
        from lib.modules.counting import Counter

        self.counter = Counter(
            objects=self.num_objects_single_image, already_sigmoided=True
        )


class VisualReasoningGqaLanguage(DomainLanguage):
    def __init__(
        self,
        language_encoding: torch.Tensor,
        visual_encoding: torch.Tensor,
        cross_encoding: torch.Tensor,
        parameters: VisualReasoningGqaParameters,
        gold_question_attentions: torch.Tensor = None,
        nmn_settings: Dict = None,
    ) -> None:
        super().__init__(start_types={Answer})
        self.object_scores = {}
        if parameters is not None:
            self.language_encoding = language_encoding
            self.visual_encoding = visual_encoding
            self.cross_encoding = cross_encoding
            self.parameters = parameters
            self.object_scores = {}
            self.nmn_settings = nmn_settings

    def log(self, prob):
        return torch.log(prob + self.parameters.epsilon)

    @predicate
    def exist(self, objects: ObjectSet) -> Answer:
        return self.greater_equal(
            self.count(objects),
            Number(mean=1, var=torch.tensor(self.parameters.fixed_variance)),
        )

    @predicate
    def relocate(self, objects: ObjectSet) -> ObjectSet:
        pass

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
        question_attention = torch.nn.functional.pad(
            question_attention.view(-1),
            pad=(0, language_encoding.shape[0] - question_attention.numel()),
        )
        attended_question = util.weighted_sum(language_encoding, question_attention)
        attended_question = attended_question.view(1, -1).repeat(
            visual_encoding.shape[0], 1
        )
        attended_objects = visual_encoding

        find_layer_inputs = torch.cat((attended_objects, attended_question), dim=-1)
        logits = self.parameters.find_layer(find_layer_inputs).view(2, -1)
        result = torch.sigmoid(logits).view(-1)
        self.object_scores[self.object_scores_index] = result
        return result

    @predicate_with_side_args(["question_attention"])
    def number(self, question_attention: torch.Tensor) -> Number:
        attended_question = util.weighted_sum(
            self.language_encoding,
            torch.nn.functional.softmax(
                self.log(question_attention) * self.numeric_mask, dim=-1
            ),
        )
        mean = self.parameters.number_layer(attended_question)
        self.question_attentions.append(question_attention.view(-1))
        return Number(mean=mean, var=torch.tensor(self.parameters.fixed_variance))

    @predicate
    def count(self, objects: ObjectSet) -> Number:
        if self.nmn_settings["use_count_module"]:
            count_probs, _ = self.parameters.counter(
                self.boxes.t().unsqueeze(0), objects.unsqueeze(0)
            )
            mean = (
                count_probs
                * torch.arange(count_probs.numel()).float().to(count_probs.device)
            ).sum()
            self.computed_count = mean
            var = self.parameters.fixed_variance
            return Number(mean=mean, var=var)
        mean = objects.sum()
        var = (objects * (1.0 - objects)).sum()
        return Number(mean=mean, var=var)

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
    def with_relation(
        self, a: ObjectSet, b: ObjectSet, question_attention: torch.Tensor
    ) -> ObjectSet:
        if self.nmn_settings["mask_non_attention"]:
            language_encoding = self.language_encoding[self.object_scores_index]
            visual_encoding = self.visual_encoding[self.object_scores_index]
            cross_encoding = self.cross_encoding[self.object_scores_index]
        else:
            language_encoding = self.language_encoding
            visual_encoding = self.visual_encoding
            cross_encoding = self.cross_encoding
        question_attention = torch.nn.functional.pad(
            question_attention.view(-1),
            pad=(0, language_encoding.shape[0] - question_attention.numel()),
        )
        attended_question = util.weighted_sum(language_encoding, question_attention)
        attended_question = attended_question.view(1, -1).repeat(
            visual_encoding.shape[0] ** 2, 1
        )
        visual_encoding_rows = visual_encoding.repeat(1, visual_encoding.shape[0]).view(
            visual_encoding.shape[0] ** 2, -1
        )
        visual_encoding_columns = visual_encoding.repeat(visual_encoding.shape[0], 1)
        relate_layer_inputs = torch.cat(
            (visual_encoding_rows, visual_encoding_columns, attended_question), dim=-1
        )
        object_pair_scores = torch.sigmoid(
            self.parameters.relate_layer(relate_layer_inputs)
        )
        object_pair_scores = object_pair_scores.view(
            visual_encoding.shape[0], visual_encoding.shape[0]
        )
        object_pair_scores2 = object_pair_scores * (
            1.0 - torch.eye(visual_encoding.shape[0]).cuda()
        )
        object_pair_scores3 = object_pair_scores2 * a.view(-1, 1).repeat(
            1, visual_encoding.shape[0]
        )
        object_pair_scores4 = object_pair_scores3 * b.view(1, -1).repeat(
            visual_encoding.shape[0], 1
        )
        object_scores = object_pair_scores4.sum(1).clamp(
            min=0.0 + self.parameters.epsilon, max=1.0 - self.parameters.epsilon
        )
        return object_scores

    @predicate_with_side_args(["question_attention"])
    def is_doing_action(
        self, objects: ObjectSet, question_attention: torch.Tensor
    ) -> ObjectSet:
        return self.filter(objects, question_attention)

    @predicate_with_side_args(["question_attention"])
    def has_orientation(
        self, objects: ObjectSet, question_attention: torch.Tensor
    ) -> ObjectSet:
        return self.filter(objects, question_attention)

    @predicate_with_side_args(["question_attention"])
    def filter(self, objects: ObjectSet, question_attention: torch.Tensor) -> ObjectSet:
        if self.nmn_settings["mask_non_attention"]:
            language_encoding = self.language_encoding[self.object_scores_index]
            visual_encoding = self.visual_encoding[self.object_scores_index]
            cross_encoding = self.cross_encoding[self.object_scores_index]
        else:
            language_encoding = self.language_encoding
            visual_encoding = self.visual_encoding
            cross_encoding = self.cross_encoding
        question_attention = torch.nn.functional.pad(
            question_attention.view(-1),
            pad=(0, language_encoding.shape[0] - question_attention.numel()),
        )
        attended_question = util.weighted_sum(language_encoding, question_attention)
        attended_question = attended_question.view(1, -1).repeat(
            visual_encoding.shape[0], 1
        )
        attended_objects = visual_encoding
        if self.nmn_settings["filter_find_same_params"]:
            attended_objects = visual_encoding.view(-1, visual_encoding.shape[-1])
            find_layer_inputs = torch.cat((attended_objects, attended_question), dim=-1)
            logits = self.parameters.find_layer(find_layer_inputs)
            filter_probs = torch.sigmoid(logits).view(-1)
        elif self.nmn_settings["use_sum_counting"]:
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
        self.object_scores[self.object_scores_index] = filter_probs
        return filter_probs * objects

    @predicate_with_side_args(["question_attention"])
    def has_attribute(
        self, objects: ObjectSet, question_attention: torch.Tensor
    ) -> ObjectSet:
        return self.filter(objects, question_attention)

    @predicate
    def all(self, a: ObjectSet, b: ObjectSet) -> Answer:
        pass

    @predicate
    def bool_and(self, a: Answer, b: Answer) -> Answer:
        return a * b

    @predicate
    def bool_or(self, a: Answer, b: Answer) -> Answer:
        return a + b - a * b

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
        if not action_sequence:
            raise ExecutionError("invalid action sequence")
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
                remaining_actions, remaining_side_args, full_seq_length=full_seq_length
            )
            arguments = []
            for _ in right_side_parts[1:]:
                (
                    argument,
                    remaining_actions,
                    remaining_side_args,
                ) = self._execute_sequence(
                    remaining_actions,
                    remaining_side_args,
                    full_seq_length=full_seq_length,
                )
                arguments.append(argument)
            self.object_scores_index = 1 + full_seq_length - len(action_sequence)
            return function(*arguments), remaining_actions, remaining_side_args
