import os
import logging
from typing import Dict, List, Tuple

import difflib
from overrides import overrides
import numpy as np
import torch
from torchvision.ops.boxes import box_iou

from allennlp.data import Vocabulary
from allennlp.data.fields.production_rule_field import ProductionRule
from allennlp.models.model import Model
from allennlp.modules import Attention, Seq2SeqEncoder, Embedding
from allennlp.modules.attention import AdditiveAttention
from allennlp.nn import util
from allennlp.semparse.domain_languages.domain_language import START_SYMBOL
from allennlp.state_machines.states import GrammarBasedState
from allennlp.state_machines.transition_functions.basic_transition_function import (
    BasicTransitionFunction,
)
from allennlp.state_machines import BeamSearch
from allennlp.state_machines.states import GrammarStatelet, RnnStatelet
from allennlp.training.metrics import Average
from allennlp.training.metrics import CategoricalAccuracy

from lib.semparse.visual_reasoning_nlvr2_language import (
    VisualReasoningNlvr2Language,
    VisualReasoningNlvr2Parameters,
)
from lib.state_machines.trainers.maximum_marginal_likelihood import (
    MaximumMarginalLikelihood,
)
from lib.modules.seq2seq_encoders.lxmert_src.lxrt.modeling import GeLU, BertLayerNorm
from lib.training.metrics.classification_module_score import ClassificationModuleScore

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("nlvr2_end_to_end_module_network")
class Nlvr2EndToEndModuleNetwork(Model):
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
    freeze_encoder: ``bool``, optional (default=True)
        If true, weights of the encoder will be frozen during training.
    dropout : ``float``, optional (default=0)
        Dropout to be applied to encoder outputs and in modules
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
    load_weights: ``str``, optional (default=None)
        Path from which to load model weights. If None or if path does not exist,
        no weights are loaded.
    use_modules: ``bool``, optional (default=True)
        If True, use modules and execute them according to programs. If False, use a
        feedforward network on top of the encoder to directly predict the label.
    positive_iou_threshold: ``float``, optional (default=0.5)
        Intersection-over-union (IOU) threshold to use for determining matches between
        ground-truth and predicted boxes in the faithfulness recall score.
    negative_iou_threshold: ``float``, optional (default=0.5)
        Intersection-over-union (IOU) threshold to use for determining matches between
        ground-truth and predicted boxes in the faithfulness precision score.
    nmn_settings: Dict, optional (default=None)
        A dictionary specifying choices determining architectures of the modules. This should
        not be None if use_modules == True.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        encoder: Seq2SeqEncoder,
        freeze_encoder: bool = False,
        dropout: float = 0.0,
        tokens_namespace: str = "tokens",
        rule_namespace: str = "rule_labels",
        denotation_namespace: str = "labels",
        num_parse_only_batches: int = 0,
        use_gold_program_for_eval: bool = True,
        load_weights: str = None,
        use_modules: bool = True,
        positive_iou_threshold: float = 0.5,
        negative_iou_threshold: float = 0.5,
        nmn_settings: Dict = None,
    ) -> None:
        super().__init__(vocab)
        self._encoder = encoder
        self._max_decoding_steps = 10
        self._add_action_bias = True
        self._dropout = torch.nn.Dropout(p=dropout)
        self._tokens_namespace = tokens_namespace
        self._rule_namespace = rule_namespace
        self._denotation_namespace = denotation_namespace
        self._denotation_accuracy = denotation_namespace
        self._num_parse_only_batches = num_parse_only_batches
        self._use_gold_program_for_eval = use_gold_program_for_eval
        self._nmn_settings = nmn_settings
        self._use_modules = use_modules
        self._training_batches_so_far = 0

        self._denotation_accuracy = CategoricalAccuracy()
        self._box_f1_score = ClassificationModuleScore(
            positive_iou_threshold=positive_iou_threshold,
            negative_iou_threshold=negative_iou_threshold,
        )
        self._best_box_f1_score = ClassificationModuleScore(
            positive_iou_threshold=positive_iou_threshold,
            negative_iou_threshold=negative_iou_threshold,
        )
        # TODO(mattg): use FullSequenceMatch instead of this.
        self._program_accuracy = Average()
        self._program_similarity = Average()
        self.loss = torch.nn.BCELoss()
        self.loss_with_logits = torch.nn.BCEWithLogitsLoss()

        self._action_padding_index = -1  # the padding value used by IndexField
        num_actions = vocab.get_vocab_size(self._rule_namespace)
        action_embedding_dim = 100
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

        if self._use_modules:
            self._language_parameters = VisualReasoningNlvr2Parameters(
                hidden_dim=self._encoder.get_output_dim(),
                initializer=self._encoder.encoder.model.init_bert_weights,
                max_boxes=self._nmn_settings["max_boxes"],
                dropout=dropout,
                nmn_settings=nmn_settings,
            )
        else:
            hid_dim = self._encoder.get_output_dim()
            self.logit_fc = torch.nn.Sequential(
                torch.nn.Linear(hid_dim * 2, hid_dim * 2),
                GeLU(),
                BertLayerNorm(hid_dim * 2, eps=1e-12),
                torch.nn.Linear(hid_dim * 2, 1),
            )
            self.logit_fc.apply(self._encoder.encoder.model.init_bert_weights)

        # This is what we pass as input in the first step of decoding, when we don't have a
        # previous action, or a previous utterance attention.
        encoder_output_dim = self._encoder.get_output_dim()

        self._decoder_num_layers = 1

        self._beam_search = BeamSearch(beam_size=10)
        self._decoder_trainer = MaximumMarginalLikelihood()
        self._first_action_embedding = torch.nn.Parameter(
            torch.FloatTensor(action_embedding_dim)
        )
        self._first_attended_utterance = torch.nn.Parameter(
            torch.FloatTensor(encoder_output_dim)
        )
        torch.nn.init.normal_(self._first_action_embedding)
        torch.nn.init.normal_(self._first_attended_utterance)
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

        # Our language is constant across instances, so we just create one up front that we can
        # re-use to construct the `GrammarStatelet`.
        self._world = VisualReasoningNlvr2Language(None, None, None, None, None, None)

        if load_weights is not None:
            if not os.path.exists(load_weights):
                print('Could not find weights path: '+load_weights+'. Continuing without loading weights.')
            else:
                if torch.cuda.is_available():
                    state = torch.load(load_weights)
                else:
                    state = torch.load(load_weights, map_location="cpu")
                encoder_prefix = "_encoder"
                lang_params_prefix = "_language_parameters"
                for key in list(state.keys()):
                    if (
                        key[: len(encoder_prefix)] != encoder_prefix
                        and key[: len(lang_params_prefix)] != lang_params_prefix
                    ):
                        del state[key]
                    if "relate_layer" in key:
                        del state[key]
                self.load_state_dict(state, strict=False)

        if freeze_encoder:
            for param in self._encoder.parameters():
                param.requires_grad = False

        self.consistency_group_map = {}

    def consistency(self, reset: bool = False):
        if reset:
            self.consistency_group_map = {}
        if len(self.consistency_group_map) == 0:
            return 0.0
        consistency = len(
            [
                group
                for group in self.consistency_group_map
                if self.consistency_group_map[group] == True
            ]
        ) / float(len(self.consistency_group_map))
        return consistency

    @overrides
    def forward(
        self,  # type: ignore
        sentence: Dict[str, torch.LongTensor],
        visual_feat: torch.Tensor,
        pos: torch.Tensor,
        image_id: List[str],
        gold_question_attentions: torch.Tensor = None,
        gold_box_annotations: List[List[List[List[float]]]] = None,
        identifier: List[str] = None,
        logical_form: List[str] = None,
        actions: List[List[ProductionRule]] = None,
        target_action_sequence: torch.LongTensor = None,
        valid_target_sequence: torch.Tensor = None,
        denotation: torch.Tensor = None,
        metadata: Dict = None,
    ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        batch_size, img_num, obj_num, feat_size = visual_feat.size()
        assert img_num == 2 and feat_size == 2048
        text_masks = util.get_text_field_mask(sentence)
        (l1, v1, text, vis_only1), x1 = self._encoder(
            sentence[self._tokens_namespace], text_masks, visual_feat[:, 0], pos[:, 0]
        )
        (l2, v2, text, vis_only2), x2 = self._encoder(
            sentence[self._tokens_namespace], text_masks, visual_feat[:, 1], pos[:, 1]
        )
        l_orig = torch.cat((l1.unsqueeze(1), l2.unsqueeze(1)), dim=1)
        v_orig = torch.cat((v1.unsqueeze(1), v2.unsqueeze(1)), dim=1)
        x_orig = torch.cat((x1.unsqueeze(1), x2.unsqueeze(1)), dim=1)
        vis_only = torch.cat((vis_only1.unsqueeze(1), vis_only2.unsqueeze(1)), dim=1)

        # NOTE: Taking the lxmert output before cross modality layer (which is the same for both images)
        # Can also try concatenating (dim=-1) the two encodings
        encoded_sentence = text

        valid_target_sequence = valid_target_sequence.long()

        initial_state = self._get_initial_state(
            encoded_sentence[valid_target_sequence == 1],
            text_masks[valid_target_sequence == 1],
            actions,
        )
        initial_state.debug_info = [[] for _ in range(batch_size)]

        if target_action_sequence is not None:
            # Remove the trailing dimension (from ListField[ListField[IndexField]]).
            target_action_sequence = target_action_sequence.squeeze(-1)
            target_action_sequence = target_action_sequence[valid_target_sequence == 1]
            target_mask = target_action_sequence != self._action_padding_index
        else:
            target_mask = None

        outputs: Dict[str, torch.Tensor] = {}
        losses = []
        if (
            self.training or self._use_gold_program_for_eval
        ) and target_action_sequence is not None:
            if valid_target_sequence.sum() > 0:
                outputs, final_states = self._decoder_trainer.decode(
                    initial_state,
                    self._transition_function,
                    (target_action_sequence.unsqueeze(1), target_mask.unsqueeze(1)),
                )

                # B X TARGET X SENTENCE
                question_attention = [
                    [
                        dbg["question_attention"]
                        for dbg in final_states[i][0].debug_info[0]
                    ]
                    for i in range(len(final_states))
                    if valid_target_sequence[i].item() == 1
                ]
                target_attn_loss = self._compute_target_attn_loss(
                    question_attention,
                    gold_question_attentions[valid_target_sequence == 1].squeeze(-1),
                )
                if not self._use_gold_program_for_eval:
                    outputs["loss"] += target_attn_loss
                else:
                    outputs["loss"] = torch.tensor(0.0)
                    if torch.cuda.is_available():
                        outputs["loss"] = outputs["loss"].cuda()
            else:
                final_states = None
                outputs["loss"] = torch.tensor(0.0).cuda()
            if (1 - valid_target_sequence).sum() > 0:
                if final_states is None:
                    final_states = {}
                initial_state = self._get_initial_state(
                    encoded_sentence[valid_target_sequence == 0],
                    text_masks[valid_target_sequence == 0],
                    actions,
                )
                remaining_states = self._beam_search.search(
                    self._max_decoding_steps,
                    initial_state,
                    self._transition_function,
                    keep_final_unfinished_states=False,
                )
                new_final_states = {}
                count = 0
                for i in range(valid_target_sequence.shape[0]):
                    if valid_target_sequence[i] < 0.5:
                        new_final_states[i] = remaining_states[i - count]
                    else:
                        new_final_states[i] = final_states[count]
                        count += 1
                final_states = new_final_states
            if final_states is None:
                len_final_states = 0
            else:
                len_final_states = len(final_states)
        else:
            initial_state = self._get_initial_state(
                encoded_sentence, text_masks, actions
            )
            final_states = self._beam_search.search(
                self._max_decoding_steps,
                initial_state,
                self._transition_function,
                keep_final_unfinished_states=False,
            )

        action_mapping = {}
        for action_index, action in enumerate(actions[0]):
            action_mapping[action_index] = action[0]

        outputs["action_mapping"] = action_mapping
        outputs["debug_info"] = []
        outputs["modules_debug_info"] = []
        outputs["best_action_sequence"] = []
        outputs["image_id"] = []
        outputs["prediction"] = []
        outputs["label"] = []
        outputs["correct"] = []
        outputs["bboxes"] = []

        outputs = self._compute_parsing_validation_outputs(
            actions,
            target_action_sequence.shape[0],
            final_states,
            initial_state,
            [
                datum
                for i, datum in enumerate(metadata)
                if valid_target_sequence[i].item() == 1
            ],
            outputs,
            target_action_sequence,
        )

        if not self._use_modules:
            logits = self.logit_fc(x.view(batch_size, -1))
            if denotation is not None:
                self._denotation_accuracy(
                    torch.cat((torch.zeros_like(logits), logits), dim=-1), denotation
                )
                if self.training:
                    outputs["loss"] += self.loss_with_logits(
                        logits.view(-1), denotation.view(-1).float()
                    )
                    self._training_batches_so_far += 1
                else:
                    outputs["loss"] = self.loss_with_logits(
                        logits.view(-1), denotation.view(-1).float()
                    )
            return outputs

        if self._nmn_settings["mask_non_attention"]:
            zero_one_mult = (
                torch.zeros_like(text_masks)
                .unsqueeze(1)
                .repeat(1, target_action_sequence.shape[1], 1)
            )
            reformatted_gold_question_attentions = torch.where(
                gold_question_attentions.squeeze(-1) == -1,
                torch.zeros_like(gold_question_attentions.squeeze(-1)),
                gold_question_attentions.squeeze(-1),
            )
            pred_question_attention = [
                torch.stack(
                    [
                        torch.nn.functional.pad(
                            dbg["question_attention"].view(-1),
                            pad=(
                                0,
                                zero_one_mult.shape[-1]
                                - dbg["question_attention"].numel(),
                            ),
                        )
                        for dbg in final_states[i][0].debug_info[0]
                    ]
                )
                for i in range(len(final_states))
            ]
            pred_question_attention = torch.stack(
                [
                    torch.nn.functional.pad(
                        attention,
                        pad=(0, 0, 0, zero_one_mult.shape[1] - attention.shape[0]),
                    )
                    for attention in pred_question_attention
                ]
            ).to(zero_one_mult.device)
            zero_one_mult.scatter_(
                2,
                reformatted_gold_question_attentions,
                torch.ones_like(reformatted_gold_question_attentions),
            )
            zero_one_mult[:, :, 0] = 1.0
            sep_indices = (
                (
                    text_masks
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
                .repeat(1, text_masks.shape[1])
                .unsqueeze(1)
                .repeat(1, target_action_sequence.shape[1], 1)
            )
            indices_dim2 = (
                torch.arange(text_masks.shape[1])
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(batch_size, target_action_sequence.shape[1], 1)
                .to(sep_indices.device)
                .long()
            )
            zero_one_mult = torch.where(
                sep_indices == indices_dim2,
                torch.ones_like(zero_one_mult),
                zero_one_mult,
            ).float()
            reshaped_questions = (
                sentence[self._tokens_namespace]
                .unsqueeze(1)
                .repeat(1, target_action_sequence.shape[1], 1)
                .view(-1, text_masks.shape[-1])
            )
            reshaped_visual_feat = (
                visual_feat.unsqueeze(1)
                .repeat(1, target_action_sequence.shape[1], 1, 1, 1)
                .view(-1, img_num, obj_num, visual_feat.shape[-1])
            )
            reshaped_pos = (
                pos.unsqueeze(1)
                .repeat(1, target_action_sequence.shape[1], 1, 1, 1)
                .view(-1, img_num, obj_num, pos.shape[-1])
            )
            zero_one_mult = zero_one_mult.view(-1, text_masks.shape[-1])
            q_att_filter = zero_one_mult.sum(1) > 2
            (l1, v1, text, vis_only1), x1 = self._encoder(
                reshaped_questions[q_att_filter, :],
                zero_one_mult[q_att_filter, :],
                reshaped_visual_feat[q_att_filter, 0, :, :],
                reshaped_pos[q_att_filter, 0, :, :],
            )
            (l2, v2, text, vis_only2), x2 = self._encoder(
                reshaped_questions[q_att_filter, :],
                zero_one_mult[q_att_filter, :],
                reshaped_visual_feat[q_att_filter, 1, :, :],
                reshaped_pos[q_att_filter, 1, :, :],
            )
            l_cat = torch.cat((l1.unsqueeze(1), l2.unsqueeze(1)), dim=1)
            v_cat = torch.cat((v1.unsqueeze(1), v2.unsqueeze(1)), dim=1)
            x_cat = torch.cat((x1.unsqueeze(1), x2.unsqueeze(1)), dim=1)
            l = [{} for _ in range(batch_size)]
            v = [{} for _ in range(batch_size)]
            x = [{} for _ in range(batch_size)]
            count = 0
            batch_index = -1
            for i in range(zero_one_mult.shape[0]):
                module_num = i % target_action_sequence.shape[1]
                if module_num == 0:
                    batch_index += 1
                    state = final_states[batch_index][0]
                    action_indices = state.action_history[0]
                    action_strings = [
                        action_mapping[action_index] for action_index in action_indices
                    ]
                if q_att_filter[i].item():
                    l[batch_index][module_num] = self._dropout(l_cat[count])
                    v[batch_index][module_num] = self._dropout(v_cat[count])
                    x[batch_index][module_num] = self._dropout(x_cat[count])
                    count += 1
        else:
            l = self._dropout(l_orig)
            v = self._dropout(v_orig)
            x = self._dropout(x_orig)

        outputs["box_acc"] = [{} for _ in range(batch_size)]
        outputs["best_box_acc"] = [{} for _ in range(batch_size)]
        outputs["box_score"] = [{} for _ in range(batch_size)]
        outputs["box_f1"] = [{} for _ in range(batch_size)]
        outputs["box_f1_overall_score"] = [{} for _ in range(batch_size)]
        outputs["best_box_f1"] = [{} for _ in range(batch_size)]
        outputs["gold_box"] = []
        outputs["ious"] = [{} for _ in range(batch_size)]

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

            # print(denotation.shape, denotation[batch_index])

            outputs["modules_debug_info"].append([])
            denotation_log_prob_list = []
            # TODO(mattg): maybe we want to limit the number of states we evaluate (programs we
            # execute) at test time, just for efficiency.
            for state_index, state in enumerate(final_states[batch_index]):
                world = VisualReasoningNlvr2Language(
                    l[batch_index],
                    v[batch_index],
                    x[batch_index],
                    self._language_parameters,
                    metadata[batch_index]["tokenized_utterance"],
                    pos[batch_index],
                    self._nmn_settings,
                )
                action_indices = state.action_history[0]
                action_strings = [
                    action_mapping[action_index] for action_index in action_indices
                ]
                # Shape: (num_denotations,)
                assert len(action_strings) == len(state.debug_info[0])
                # Plug in gold question attentions
                for i in range(len(state.debug_info[0])):
                    if (
                        self._use_gold_program_for_eval
                        and valid_target_sequence[batch_index] == 1
                    ):
                        n_att_words = (
                            (gold_question_attentions[batch_index, i] >= 0)
                            .float()
                            .sum()
                        )
                        state.debug_info[0][i]["question_attention"] = torch.zeros_like(
                            state.debug_info[0][i]["question_attention"]
                        )
                        if n_att_words > 0:
                            for j in gold_question_attentions[batch_index, i]:
                                if j >= 0:
                                    state.debug_info[0][i]["question_attention"][j] = (
                                        1.0 / n_att_words
                                    )
                    if (
                        i not in l[batch_index]
                        and self._nmn_settings["mask_non_attention"]
                        and (
                            action_strings[i][-4:] == "find"
                            or action_strings[i][-6:] == "filter"
                            or action_strings[i][-13:] == "with_relation"
                            or action_strings[i][-7:] == "project"
                        )
                    ):
                        l[batch_index][i] = l_orig[batch_index, :, :]
                        v[batch_index][i] = v_orig[batch_index, :, :]
                        x[batch_index][i] = x_orig[batch_index, :]
                        world = VisualReasoningNlvr2Language(
                            l[batch_index],
                            v[batch_index],
                            x[batch_index],
                            self._language_parameters,
                            metadata[batch_index]["tokenized_utterance"],
                            pos[batch_index],
                            self._nmn_settings,
                        )
                world.parameters.train(self.training)

                state_denotation_probs = world.execute_action_sequence(
                    action_strings, state.debug_info[0]
                )
                outputs["modules_debug_info"][batch_index].append(
                    world.modules_debug_info
                )

                # P(denotation | parse) * P(parse | question)
                world_log_prob = (state_denotation_probs + 1e-6).log()
                if not self._use_gold_program_for_eval:
                    world_log_prob += state.score[0]
                denotation_log_prob_list.append(world_log_prob)

            # P(denotation | parse) * P(parse | question) for the all programs on the beam.
            # Shape: (beam_size, num_denotations)
            denotation_log_probs = torch.stack(denotation_log_prob_list)
            # \Sum_parse P(denotation | parse) * P(parse | question) = P(denotation | question)
            # Shape: (num_denotations,)
            marginalized_denotation_log_probs = util.logsumexp(
                denotation_log_probs, dim=0
            )
            if denotation is not None:
                # This line is needed, otherwise we have numbers slightly exceeding 0..1. Should check why
                state_denotation_probs = state_denotation_probs.clamp(min=0, max=1)

                loss = self.loss(
                    state_denotation_probs.unsqueeze(0),
                    denotation[batch_index].unsqueeze(0).float(),
                ).view(1)
                losses.append(loss)
                self._denotation_accuracy(
                    torch.tensor(
                        [1 - state_denotation_probs, state_denotation_probs]
                    ).to(denotation.device),
                    denotation[batch_index],
                )
                group_id = metadata[batch_index]["identifier"].split("-")
                group_id = group_id[0] + "-" + group_id[1] + "-" + group_id[-1]
                if group_id not in self.consistency_group_map:
                    self.consistency_group_map[group_id] = True
                if (
                    state_denotation_probs.item() >= 0.5
                    and denotation[batch_index].item() < 0.5
                ) or (
                    state_denotation_probs.item() < 0.5
                    and denotation[batch_index].item() > 0.5
                ):
                    self.consistency_group_map[group_id] = False
                if (
                    gold_box_annotations is not None
                    and len(gold_box_annotations[batch_index]) > 0
                ):
                    box_f1_score, overall_f1_score_value = self._box_f1_score(
                        outputs["modules_debug_info"][batch_index][0],
                        gold_box_annotations[batch_index],
                        pos[batch_index],
                    )
                    outputs["box_f1"][batch_index] = box_f1_score
                    outputs["box_f1_overall_score"][
                        batch_index
                    ] = overall_f1_score_value
                    best_f1_predictions = self._best_box_f1_score.compute_best_box_predictions(
                        outputs["modules_debug_info"][batch_index][0],
                        gold_box_annotations[batch_index],
                        pos[batch_index],
                    )
                    best_f1, _ = self._best_box_f1_score(
                        best_f1_predictions,
                        gold_box_annotations[batch_index],
                        pos[batch_index],
                    )
                    outputs["best_box_f1"][batch_index] = best_f1
                    outputs["gold_box"].append(gold_box_annotations[batch_index])
            outputs["image_id"].append(image_id[batch_index])
            outputs["prediction"].append(world_log_prob.exp())
            if denotation is not None:
                outputs["label"].append(denotation[batch_index])
                outputs["correct"].append(
                    world_log_prob.exp().round().int() == denotation[batch_index].int()
                )
            outputs["bboxes"].append(pos[batch_index])
        if losses:
            outputs["loss"] += torch.stack(losses).mean()
        if self.training:
            self._training_batches_so_far += 1
        return outputs

    def _compute_parsing_validation_outputs(
        self,
        actions,
        batch_size,
        final_states,
        initial_state,
        metadata,
        outputs,
        target_action_sequence,
    ):
        if (
            not self.training
            and target_action_sequence is not None
            and target_action_sequence.numel() > 0
        ):
            outputs["parse_correct"] = []

            # skip beam search if we already searched
            # if self._use_gold_program_for_eval:
            #     final_states = self._beam_search.search(self._max_decoding_steps,
            #                                             initial_state,
            #                                             self._transition_function,
            #                                             keep_final_unfinished_states=False)
            best_action_sequences: Dict[int, List[List[int]]] = {}
            for i in range(batch_size):
                # Decoding may not have terminated with any completed logical forms, if `num_steps`
                # isn't long enough (or if the model is not trained enough and gets into an
                # infinite action loop).
                if i in final_states:
                    best_action_indices = final_states[i][0].action_history[0]
                    best_action_sequences[i] = best_action_indices

                    targets = target_action_sequence[i].data
                    sequence_in_targets = self._action_history_match(
                        best_action_indices, targets
                    )
                    self._program_accuracy(sequence_in_targets)

                    similarity = difflib.SequenceMatcher(
                        None, best_action_indices, targets
                    )
                    self._program_similarity(similarity.ratio())
                    outputs["parse_correct"].append(sequence_in_targets)
                else:
                    self._program_accuracy(0)
                    self._program_similarity(0)
                    continue

            batch_action_strings = self._get_action_strings(
                actions, best_action_sequences
            )
            if metadata is not None:
                outputs["sentence_tokens"] = [
                    x["tokenized_utterance"] for x in metadata
                ]
                outputs["utterance"] = [x["utterance"] for x in metadata]
                outputs["parse_gold"] = [x["gold"] for x in metadata]
            outputs["debug_info"] = []
            outputs["parse_predicted"] = []
            outputs["action_mapping"] = []
            for i in range(batch_size):
                if i in final_states:
                    outputs["debug_info"].append(final_states[i][0].debug_info[0])  # type: ignore
                    outputs["action_mapping"].append([a[0] for a in actions[i]])
                    outputs["parse_predicted"].append(
                        self._world.action_sequence_to_logical_form(
                            batch_action_strings[i]
                        )
                    )
            outputs["best_action_strings"] = batch_action_strings
            action_mapping = {}
            for batch_index, batch_actions in enumerate(actions):
                for action_index, action in enumerate(batch_actions):
                    action_mapping[(batch_index, action_index)] = action[0]

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
        metrics = {
            "denotation_acc": self._denotation_accuracy.get_metric(reset),
            "program_acc": self._program_accuracy.get_metric(reset),
            "consistency": self.consistency(reset),
        }
        for m in self._box_f1_score.modules:
            metrics["_" + m + "_box_f1"] = self._box_f1_score.get_metric(
                reset=False, module=m
            )["f1"]
            metrics["_" + m + "_best_box_f1"] = self._best_box_f1_score.get_metric(
                reset=False, module=m
            )["f1"]
        box_f1_pr = self._box_f1_score.get_metric(reset=reset)
        for key in box_f1_pr:
            metrics["overall_box_" + key] = box_f1_pr[key]
        best_box_f1_pr = self._best_box_f1_score.get_metric(reset=reset)
        for key in best_box_f1_pr:
            metrics["overall_best_box_" + key] = best_box_f1_pr[key]
        return metrics

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

    def _compute_target_attn_loss(self, question_attention, gold_question_attention):
        attn_loss = 0
        normalizer = 0

        gold_question_attention = gold_question_attention.cpu().numpy()

        # TODO: Pad and batch this for performance
        for instance_attn, gld_instance_attn in zip(
            question_attention, gold_question_attention
        ):
            for step_attn, gld_step_attn in zip(instance_attn, gld_instance_attn):
                if gld_step_attn[0] == -1:
                    continue
                # consider only non-padding indices
                gld_step_attn = [a for a in gld_step_attn if a > -1]
                given_attn = step_attn[gld_step_attn]
                attn_loss += (given_attn.sum() + 1e-8).log()
                normalizer += 1

        if normalizer == 0:
            return 0
        else:
            return -1 * (attn_loss / normalizer)

    @classmethod
    def _get_action_strings(
        cls,
        possible_actions: List[List[ProductionRule]],
        action_indices: Dict[int, List[List[int]]],
    ) -> List[List[List[str]]]:
        """
        Takes a list of possible actions and indices of decoded actions into those possible actions
        for a batch and returns sequences of action strings. We assume ``action_indices`` is a dict
        mapping batch indices to k-best decoded sequence lists.
        """
        all_action_strings: List[List[List[str]]] = []
        batch_size = len(possible_actions)
        for i in range(batch_size):
            batch_actions = possible_actions[i]
            batch_best_sequences = action_indices[i] if i in action_indices else []
            # This will append an empty list to ``all_action_strings`` if ``batch_best_sequences``
            # is empty.
            action_strings = [
                batch_actions[rule_id][0] for rule_id in batch_best_sequences
            ]
            all_action_strings.append(action_strings)
        return all_action_strings
