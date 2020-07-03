import logging
from typing import Any, Dict, List, Tuple

import difflib
import sqlparse
import json
from overrides import overrides
import numpy as np
import torch

from allennlp.common.util import pad_sequence_to_length
from allennlp.data import Vocabulary
from allennlp.data.fields.production_rule_field import ProductionRule
from allennlp.semparse.executors import SqlExecutor
from allennlp.models.model import Model
from allennlp.modules import Attention, Seq2SeqEncoder, TextFieldEmbedder, Embedding
from allennlp.modules.attention import AdditiveAttention
from allennlp.nn import util
from allennlp.semparse.domain_languages.domain_language import START_SYMBOL
from allennlp.state_machines.states import GrammarBasedState
from allennlp.state_machines.transition_functions.basic_transition_function import (
    BasicTransitionFunction,
)
from allennlp.state_machines import BeamSearch, ConstrainedBeamSearch
from allennlp.state_machines.trainers import MaximumMarginalLikelihood
from allennlp.state_machines.states import GrammarStatelet, RnnStatelet
from allennlp.training.metrics import Average
from allennlp.training.metrics import CategoricalAccuracy

from lib.semparse.visual_reasoning_gqa_language import (
    VisualReasoningGqaLanguage,
    VisualReasoningGqaParameters,
)

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("gqa_end_to_end_module_network")
class GqaEndToEndModuleNetwork(Model):
    """
    A re-implementation of `End-to-End Module Networks for Visual Question Answering
    <https://www.semanticscholar.org/paper/Learning-to-Reason%3A-End-to-End-Module-Networks-for-Hu-Andreas/5e07d6951b7bc0c4113313a9586ce8178eacdf57>`_

    This implementation is based on our semantic parsing framework, and uses marginal likelihood to
    train the parser when labeled action sequences are not available.  It is `not` an exact
    re-implementation, but rather a very similar model with some significant differences in how the
    grammar is used.

    Parameters
    ----------
    vocab : ``Vocabulary``
    encoder : ``Seq2SeqEncoder``
        The encoder to use for the input utterance.
    dropout : ``float``, optional (default=0)
        Dropout to be applied to encoder outputs and in modules
    object_loss_multiplier: ``float``, optional (default=0)
        Multiplier used for the auxiliary loss on module outputs that can be computed for
        the find and filter modules if the module-wise annotations are provided.
    denotation_loss_multiplier: ``float``, optional (default=1)
        Multiplier used for the denotation loss (i.e. loss for predicting the label).
    tokens_namespace : ``str``, optional (default=tokens)
        The vocabulary namespace to use for tokens.  The default corresponds to the
        default used in the dataset reader, so you likely don't need to modify this.
    rule_namespace : ``str``, optional (default=rule_labels)
        The vocabulary namespace to use for production rules.  The default corresponds to the
        default used in the dataset reader, so you likely don't need to modify this.
    denotation_namespace : ``str``, optional (default=labels)
        The vocabulary namespace to use for output labels.  The default corresponds to the
        default used in the dataset reader, so you likely don't need to modify this.
    num_parse_only_batches : ``int``, optional (default=0)
        We will use this many training batches of `only` parse supervision, not denotation
        supervision.  This is helpful in cases where learning the correct programs at the same time
        as learning the program executor, both from scratch is challenging.  This only works if you
        have labeled programs.
    use_gold_program_for_eval : ``bool``, optional (default=True)
        If true, we will use the gold program for evaluation when it is available (this only tests
        the program executor, not the parser).
    nmn_settings: Dict, optional (default=None)
        A dictionary specifying choices determining architectures of the modules. This should
        not be None if use_modules == True.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        encoder: Seq2SeqEncoder,
        dropout: float = 0.0,
        object_loss_multiplier: float = 0.0,
        denotation_loss_multiplier: float = 1.0,
        tokens_namespace: str = "tokens",
        rule_namespace: str = "rule_labels",
        denotation_namespace: str = "labels",
        num_parse_only_batches: int = 0,
        use_gold_program_for_eval: bool = False,
        nmn_settings: Dict = None,
    ) -> None:
        # Atis semantic parser init
        super().__init__(vocab)
        self._encoder = encoder
        self._dropout = torch.nn.Dropout(p=dropout)
        self._obj_loss_multiplier = object_loss_multiplier
        self._denotation_loss_multiplier = denotation_loss_multiplier
        self._tokens_namespace = tokens_namespace
        self._rule_namespace = rule_namespace
        self._denotation_namespace = denotation_namespace
        self._num_parse_only_batches = num_parse_only_batches
        self._use_gold_program_for_eval = use_gold_program_for_eval
        self._nmn_settings = nmn_settings
        self._training_batches_so_far = 0

        self._denotation_accuracy = CategoricalAccuracy()
        self._proposal_accuracy = CategoricalAccuracy()
        # TODO(mattg): use FullSequenceMatch instead of this.
        self._program_accuracy = Average()
        self.loss = torch.nn.BCELoss()

        self._action_padding_index = -1  # the padding value used by IndexField
        num_actions = vocab.get_vocab_size(self._rule_namespace)
        action_embedding_dim = 100
        self._add_action_bias = True
        if self._add_action_bias:
            input_action_dim = action_embedding_dim + 1
        else:
            input_action_dim = action_embedding_dim
        self._action_embedder = Embedding(
            num_embeddings=num_actions, embedding_dim=input_action_dim
        )
        self._output_action_embedder = Embedding(
            num_embeddings=num_actions, embedding_dim=action_embedding_dim
        )

        self._language_parameters = VisualReasoningGqaParameters(
            hidden_dim=self._encoder.get_output_dim(),
            initializer=self._encoder.encoder.model.init_bert_weights,
        )

        # This is what we pass as input in the first step of decoding, when we don't have a
        # previous action, or a previous utterance attention.
        self._first_action_embedding = torch.nn.Parameter(
            torch.FloatTensor(action_embedding_dim)
        )
        # encoder_output_dim = self._lxrt_encoder.get_output_dim()
        encoder_output_dim = self._encoder.get_output_dim()
        self._first_attended_utterance = torch.nn.Parameter(
            torch.FloatTensor(encoder_output_dim)
        )
        torch.nn.init.normal_(self._first_action_embedding)
        torch.nn.init.normal_(self._first_attended_utterance)

        self._decoder_num_layers = 1

        self._beam_search = BeamSearch(beam_size=10)
        self._decoder_trainer = MaximumMarginalLikelihood()
        self._transition_function = BasicTransitionFunction(
            encoder_output_dim=encoder_output_dim,
            action_embedding_dim=action_embedding_dim,
            input_attention=AdditiveAttention(
                vector_dim=encoder_output_dim, matrix_dim=encoder_output_dim
            ),
            add_action_bias=self._add_action_bias,
            dropout=dropout,
            num_layers=self._decoder_num_layers,
        )
        self._language_parameters.apply(self._encoder.encoder.model.init_bert_weights)
        # attention.apply(self._lxrt_encoder.encoder.model.init_bert_weights)
        # self._transition_function.apply(self._lxrt_encoder.encoder.model.init_bert_weights)

        # Our language is constant across instances, so we just create one up front that we can
        # re-use to construct the `GrammarStatelet`.
        self._world = VisualReasoningGqaLanguage(None, None, None, None, None)

    @overrides
    def forward(
        self,  # type: ignore
        question_field: Dict[str, torch.LongTensor],
        visual_feat: torch.Tensor,
        pos: torch.Tensor,
        image_id: List[str],
        gold_question_attentions: torch.Tensor = None,
        identifier: List[str] = None,
        logical_form: List[str] = None,
        actions: List[List[ProductionRule]] = None,
        target_action_sequence: torch.LongTensor = None,
        gold_object_choices: torch.Tensor = None,
        denotation: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        batch_size, obj_num, feat_size = visual_feat.size()
        assert obj_num == 36 and feat_size == 2048
        text_masks = util.get_text_field_mask(question_field)
        (l_orig, v_orig, text, vis_only), x_orig = self._encoder(
            question_field[self._tokens_namespace], text_masks, visual_feat, pos
        )

        text_masks = text_masks.float()
        # NOTE: Taking the lxmert output before cross modality layer (which is the same for both images)
        # Can also try concatenating (dim=-1) the two encodings
        encoded_sentence = text

        initial_state = self._get_initial_state(encoded_sentence, text_masks, actions)
        initial_state.debug_info = [[] for _ in range(batch_size)]

        if target_action_sequence is not None:
            # Remove the trailing dimension (from ListField[ListField[IndexField]]).
            target_action_sequence = target_action_sequence.squeeze(-1)
            target_mask = target_action_sequence != self._action_padding_index
        else:
            target_mask = None

        outputs: Dict[str, torch.Tensor] = {}
        losses = []
        if (
            self.training or self._use_gold_program_for_eval
        ) and target_action_sequence is not None:
            # target_action_sequence is of shape (batch_size, 1, sequence_length) here after we
            # unsqueeze it for the MML trainer.
            search = ConstrainedBeamSearch(
                beam_size=None,
                allowed_sequences=target_action_sequence.unsqueeze(1),
                allowed_sequence_mask=target_mask.unsqueeze(1),
            )
            final_states = search.search(initial_state, self._transition_function)
            if self._training_batches_so_far < self._num_parse_only_batches:
                for batch_index in range(batch_size):
                    if not final_states[batch_index]:
                        logger.error(f"No pogram found for batch index {batch_index}")
                        continue
                    losses.append(-final_states[batch_index][0].score[0])
        else:
            final_states = self._beam_search.search(
                self._max_decoding_steps,
                initial_state,
                self._transition_function,
                keep_final_unfinished_states=False,
            )

        action_mapping = {}
        for action_index, action in enumerate(actions[0]):
            action_mapping[action_index] = action[0]

        outputs: Dict[str, Any] = {"action_mapping": action_mapping}
        outputs["best_action_sequence"] = []
        outputs["debug_info"] = []

        if self._nmn_settings["mask_non_attention"]:
            zero_one_mult = torch.zeros_like(gold_question_attentions)
            zero_one_mult.copy_(gold_question_attentions)
            zero_one_mult[:, :, 0] = 1.0
            # sep_indices = text_masks.argmax(1).long()
            sep_indices = (
                (
                    text_masks.long()
                    * (
                        1
                        + torch.arange(text_masks.shape[1])
                        .unsqueeze(0)
                        .repeat(batch_size, 1)
                        .to(text_masks.device)
                    )
                )
                .argmax(1)
                .long()
            )
            sep_indices = (
                sep_indices.unsqueeze(1)
                .repeat(1, gold_question_attentions.shape[2])
                .unsqueeze(1)
                .repeat(1, gold_question_attentions.shape[1], 1)
            )
            indices_dim2 = (
                torch.arange(gold_question_attentions.shape[2])
                .unsqueeze(0)
                .repeat(
                    gold_question_attentions.shape[0],
                    gold_question_attentions.shape[1],
                    1,
                )
                .to(sep_indices.device)
                .long()
            )
            zero_one_mult = torch.where(
                sep_indices == indices_dim2,
                torch.ones_like(zero_one_mult),
                zero_one_mult,
            ).float()
            reshaped_questions = (
                question_field[self._tokens_namespace]
                .unsqueeze(1)
                .repeat(1, gold_question_attentions.shape[1], 1)
                .view(-1, gold_question_attentions.shape[-1])
            )
            reshaped_visual_feat = (
                visual_feat.unsqueeze(1)
                .repeat(1, gold_question_attentions.shape[1], 1, 1)
                .view(-1, obj_num, visual_feat.shape[-1])
            )
            reshaped_pos = (
                pos.unsqueeze(1)
                .repeat(1, gold_question_attentions.shape[1], 1, 1)
                .view(-1, obj_num, pos.shape[-1])
            )
            zero_one_mult = zero_one_mult.view(-1, gold_question_attentions.shape[-1])
            q_att_filter = zero_one_mult.sum(1) > 2
            (l_relevant, v_relevant, _, _), x_relevant = self._encoder(
                reshaped_questions[q_att_filter, :],
                zero_one_mult[q_att_filter, :],
                reshaped_visual_feat[q_att_filter, :, :],
                reshaped_pos[q_att_filter, :, :],
            )
            l = [{} for _ in range(batch_size)]
            v = [{} for _ in range(batch_size)]
            x = [{} for _ in range(batch_size)]
            count = 0
            batch_index = -1
            for i in range(zero_one_mult.shape[0]):
                module_num = i % target_action_sequence.shape[1]
                if module_num == 0:
                    batch_index += 1
                if q_att_filter[i].item():
                    l[batch_index][module_num] = l_relevant[count]
                    v[batch_index][module_num] = v_relevant[count]
                    x[batch_index][module_num] = x_relevant[count]
                    count += 1
        else:
            l = l_orig
            v = v_orig
            x = x_orig

        for batch_index in range(batch_size):
            if (
                self.training
                and self._training_batches_so_far < self._num_parse_only_batches
            ):
                continue
            if not final_states[batch_index]:
                logger.error(f"No pogram found for batch index {batch_index}")
                outputs["best_action_sequence"].append([])
                outputs["debug_info"].append([])
                continue
            world = VisualReasoningGqaLanguage(
                l[batch_index],
                v[batch_index],
                x[batch_index],
                # initial_state.rnn_state[batch_index].encoder_outputs[batch_index],
                self._language_parameters,
                pos[batch_index],
                nmn_settings=self._nmn_settings,
            )

            denotation_log_prob_list = []
            # TODO(mattg): maybe we want to limit the number of states we evaluate (programs we
            # execute) at test time, just for efficiency.
            for state_index, state in enumerate(final_states[batch_index]):
                action_indices = state.action_history[0]
                action_strings = [
                    action_mapping[action_index] for action_index in action_indices
                ]
                # Shape: (num_denotations,)
                assert len(action_strings) == len(state.debug_info[0])
                # Plug in gold question attentions
                for i in range(len(state.debug_info[0])):
                    if gold_question_attentions[batch_index, i, :].sum() > 0:
                        state.debug_info[0][i]["question_attention"] = (
                            gold_question_attentions[batch_index, i, :].float()
                            / gold_question_attentions[batch_index, i, :].sum()
                        )
                    elif self._nmn_settings["mask_non_attention"] and (
                        action_strings[i][-4:] == "find"
                        or action_strings[i][-6:] == "filter"
                        or action_strings[i][-13:] == "with_relation"
                    ):
                        state.debug_info[0][i]["question_attention"] = (
                            torch.ones_like(
                                gold_question_attentions[batch_index, i, :]
                            ).float()
                            / gold_question_attentions[batch_index, i, :].numel()
                        )
                        l[batch_index][i] = l_orig[batch_index]
                        v[batch_index][i] = v_orig[batch_index]
                        x[batch_index][i] = x_orig[batch_index]
                        world = VisualReasoningGqaLanguage(
                            l[batch_index],
                            v[batch_index],
                            x[batch_index],
                            # initial_state.rnn_state[batch_index].encoder_outputs[batch_index],
                            self._language_parameters,
                            pos[batch_index],
                            nmn_settings=self._nmn_settings,
                        )
                # print(action_strings)
                state_denotation_log_probs = world.execute_action_sequence(
                    action_strings, state.debug_info[0]
                )
                # prob2 = world.execute_action_sequence(action_strings, state.debug_info[0])

                # P(denotation | parse) * P(parse | question)
                denotation_log_prob_list.append(state_denotation_log_probs)

                if not self._use_gold_program_for_eval:
                    denotation_log_prob_list[-1] += state.score[0]
                if state_index == 0:
                    outputs["best_action_sequence"].append(action_strings)
                    outputs["debug_info"].append(state.debug_info[0])
                    if target_action_sequence is not None:
                        targets = target_action_sequence[batch_index].data
                        program_correct = self._action_history_match(
                            action_indices, targets
                        )
                        self._program_accuracy(program_correct)

            # P(denotation | parse) * P(parse | question) for the all programs on the beam.
            # Shape: (beam_size, num_denotations)
            denotation_log_probs = torch.stack(denotation_log_prob_list)
            # \Sum_parse P(denotation | parse) * P(parse | question) = P(denotation | question)
            # Shape: (num_denotations,)
            marginalized_denotation_log_probs = util.logsumexp(
                denotation_log_probs, dim=0
            )
            if denotation is not None:
                loss = (
                    self.loss(
                        state_denotation_log_probs.unsqueeze(0),
                        denotation[batch_index].unsqueeze(0).float(),
                    ).view(1)
                    * self._denotation_loss_multiplier
                )
                losses.append(loss)
                self._denotation_accuracy(
                    torch.tensor(
                        [1 - state_denotation_log_probs, state_denotation_log_probs]
                    ).to(denotation.device),
                    denotation[batch_index],
                )
                if gold_object_choices is not None:
                    gold_objects = gold_object_choices[batch_index, :, :]
                    predicted_objects = torch.zeros_like(gold_objects)
                    for index in world.object_scores:
                        predicted_objects[index, :] = world.object_scores[index]
                    obj_exists = gold_objects.max(1)[0] > 0
                    # Only look at modules where at least one of the proposals has the object of interest
                    predicted_objects = predicted_objects[obj_exists, :]
                    gold_objects = gold_objects[obj_exists, :]
                    gold_objects = gold_objects.view(-1)
                    predicted_objects = predicted_objects.view(-1)
                    if gold_objects.numel() > 0:
                        loss += self._obj_loss_multiplier * self.loss(
                            predicted_objects, (gold_objects.float() + 1) / 2
                        )
                        self._proposal_accuracy(
                            torch.cat(
                                (
                                    1.0 - predicted_objects.view(-1, 1),
                                    predicted_objects.view(-1, 1),
                                ),
                                dim=-1,
                            ),
                            (gold_objects + 1) / 2,
                        )
        if losses:
            outputs["loss"] = torch.stack(losses).mean()
        if self.training:
            self._training_batches_so_far += 1
        return outputs

    def _get_initial_state(
        self,
        encoder_outputs: torch.Tensor,
        utterance_mask: torch.Tensor,
        actions: List[ProductionRule],
    ) -> GrammarBasedState:
        batch_size = encoder_outputs.size(0)

        # This will be our initial hidden state and memory cell for the decoder LSTM.
        final_encoder_output = util.get_final_encoder_states(
            encoder_outputs, utterance_mask, self._encoder.is_bidirectional()
        )
        # Use CLS states as final encoder outputs
        # final_encoder_output = encoder_outputs[:,0,:]
        memory_cell = encoder_outputs.new_zeros(batch_size, encoder_outputs.shape[-1])
        initial_score = encoder_outputs.data.new_zeros(batch_size)
        attended_sentence, _ = self._transition_function.attend_on_question(
            final_encoder_output, encoder_outputs, utterance_mask
        )

        # To make grouping states together in the decoder easier, we convert the batch dimension in
        # all of our tensors into an outer list.  For instance, the encoder outputs have shape
        # `(batch_size, utterance_length, encoder_output_dim)`.  We need to convert this into a list
        # of `batch_size` tensors, each of shape `(utterance_length, encoder_output_dim)`.  Then we
        # won't have to do any index selects, or anything, we'll just do some `torch.cat()`s.
        initial_score_list = [initial_score[i] for i in range(batch_size)]
        encoder_output_list = [encoder_outputs[i] for i in range(batch_size)]
        utterance_mask_list = [utterance_mask[i] for i in range(batch_size)]
        initial_rnn_state = []
        for i in range(batch_size):
            if self._decoder_num_layers > 1:
                encoder_output = final_encoder_output[i].repeat(
                    self._decoder_num_layers, 1
                )
                cell = memory_cell[i].repeat(self._decoder_num_layers, 1)
            else:
                encoder_output = final_encoder_output[i]
                cell = memory_cell[i]
            initial_rnn_state.append(
                RnnStatelet(
                    encoder_output,
                    cell,
                    self._first_action_embedding,
                    attended_sentence[i],
                    encoder_output_list,
                    utterance_mask_list,
                )
            )

        initial_grammar_state = [
            self._create_grammar_state(actions[i]) for i in range(batch_size)
        ]

        initial_state = GrammarBasedState(
            batch_indices=list(range(batch_size)),
            action_history=[[] for _ in range(batch_size)],
            score=initial_score_list,
            rnn_state=initial_rnn_state,
            grammar_state=initial_grammar_state,
            possible_actions=actions,
            debug_info=[[] for _ in range(batch_size)],
        )
        return initial_state

    @staticmethod
    def _action_history_match(predicted: List[int], targets: torch.LongTensor) -> int:
        # TODO(mattg): this could probably be moved into a FullSequenceMatch metric, or something.
        # Check if target is big enough to cover prediction (including start/end symbols)
        if len(predicted) > targets.size(0):
            return 0
        predicted_tensor = targets.new_tensor(predicted)
        targets_trimmed = targets[: len(predicted)]
        # Return 1 if the predicted sequence is anywhere in the list of targets.
        return predicted_tensor.equal(targets_trimmed)

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            "denotation_acc": self._denotation_accuracy.get_metric(reset),
            "program_acc": self._program_accuracy.get_metric(reset),
            # 'question_att_acc': self._question_att_accuracy.get_metric(reset),
        }

    def _create_grammar_state(
        self, possible_actions: List[ProductionRule]
    ) -> GrammarStatelet:
        """
        This method creates the GrammarStatelet object that's used for decoding.  Part of creating
        that is creating the `valid_actions` dictionary, which contains embedded representations of
        all of the valid actions.  So, we create that here as well.

        The inputs to this method are for a `single instance in the batch`; none of the tensors we
        create here are batched.  We grab the global action ids from the input ``ProductionRules``,
        and we use those to embed the valid actions for every non-terminal type.

        Parameters
        ----------
        possible_actions : ``List[ProductionRule]``
            From the input to ``forward`` for a single batch instance.
        """
        action_map = {}
        for action_index, action in enumerate(possible_actions):
            action_string = action[0]
            action_map[action_string] = action_index

        valid_actions = self._world.get_nonterminal_productions()

        translated_valid_actions: Dict[
            str, Dict[str, Tuple[torch.Tensor, torch.Tensor, List[int]]]
        ] = {}
        for key, action_strings in valid_actions.items():
            translated_valid_actions[key] = {}
            # `key` here is a non-terminal from the grammar, and `action_strings` are all the valid
            # productions of that non-terminal.  We'll first split those productions by global vs.
            # linked action.

            action_indices = [
                action_map[action_string] for action_string in action_strings
            ]
            production_rule_arrays = [
                (possible_actions[index], index) for index in action_indices
            ]
            global_actions = []
            for production_rule_array, action_index in production_rule_arrays:
                global_actions.append((production_rule_array[2], action_index))

            global_action_tensors, global_action_ids = zip(*global_actions)
            global_action_tensor = torch.cat(global_action_tensors, dim=0).long()
            global_input_embeddings = self._action_embedder(global_action_tensor)
            global_output_embeddings = self._output_action_embedder(
                global_action_tensor
            )
            translated_valid_actions[key]["global"] = (
                global_input_embeddings,
                global_output_embeddings,
                list(global_action_ids),
            )

        return GrammarStatelet(
            [START_SYMBOL], translated_valid_actions, self._world.is_nonterminal
        )

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
        time, to finalize predictions.  This is (confusingly) a separate notion from the "decoder"
        in "encoder/decoder", where that decoder logic lives in ``TransitionFunction``.

        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called ``predicted_actions`` to the ``output_dict``.
        """
        # TODO(mattg): FIX THIS - I haven't touched this method yet.
        action_mapping = output_dict["action_mapping"]
        best_actions = output_dict["best_action_sequence"]
        debug_infos = output_dict["debug_info"]
        batch_action_info = []
        for batch_index, (predicted_actions, debug_info) in enumerate(
            zip(best_actions, debug_infos)
        ):
            instance_action_info = []
            for predicted_action, action_debug_info in zip(
                predicted_actions, debug_info
            ):
                action_info = {}
                action_info["predicted_action"] = predicted_action
                considered_actions = action_debug_info["considered_actions"]
                probabilities = action_debug_info["probabilities"]
                actions = []
                for action, probability in zip(considered_actions, probabilities):
                    if action != -1:
                        actions.append(
                            (action_mapping[(batch_index, action)], probability)
                        )
                actions.sort()
                considered_actions, probabilities = zip(*actions)
                action_info["considered_actions"] = considered_actions
                action_info["action_probabilities"] = probabilities
                action_info["utterance_attention"] = action_debug_info.get(
                    "question_attention", []
                )
                instance_action_info.append(action_info)
            batch_action_info.append(instance_action_info)
        output_dict["predicted_actions"] = batch_action_info
        return output_dict
