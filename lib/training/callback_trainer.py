"""
The ``CallbackTrainer`` should be considered experimental code.
Its API may change at any time, and it may disappear altogether.
"""
from typing import List

import torch
from allennlp.data.iterators.data_iterator import TensorDict
from allennlp.nn import util as nn_util
from allennlp.training import util as training_util, CallbackTrainer
from allennlp.training.trainer_base import TrainerBase
from overrides import overrides


@TrainerBase.register("callback_save_predictions")
class CallbackTrainerSavePredictions(CallbackTrainer):
    @overrides
    def batch_loss(
        self,
        batch_group: List[TensorDict],
        for_training: bool,
        return_all_outputs=False,
    ) -> torch.Tensor:
        """
        Does a forward pass on the given batches and returns the ``loss`` value in the result.
        If ``for_training`` is `True` also applies regularization penalty.

        This is a method on the trainer so that it can be used both in training and validation
        (which are handled separately).
        """
        if self._multiple_gpu:
            output_dict = training_util.data_parallel(
                batch_group, self.model, self._cuda_devices
            )
        else:
            assert len(batch_group) == 1
            batch = batch_group[0]
            batch = nn_util.move_to_device(batch, self._cuda_devices[0])
            output_dict = self.model(**batch)

        try:
            loss = output_dict["loss"]
            if for_training:
                loss += self.model.get_regularization_penalty()
        except KeyError:
            if for_training:
                raise RuntimeError(
                    "The model you are trying to optimize does not contain a"
                    " 'loss' key in the output of model.forward(inputs)."
                )
            loss = None

        if return_all_outputs:
            return loss, output_dict
        else:
            return loss
