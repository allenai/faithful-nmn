import gzip
import json
import os
from typing import Iterable, List, TYPE_CHECKING, Dict
import logging
import math

import torch
from allennlp.common.checks import ConfigurationError

from allennlp.common.tqdm import Tqdm
from allennlp.common.util import lazy_groups_of, sanitize
from allennlp.data.instance import Instance
from allennlp.data.iterators import DataIterator
from allennlp.training import util as training_util
from allennlp.training.callbacks import Validate
from allennlp.training.callbacks.callback import Callback, handle_event
from allennlp.training.callbacks.events import Events

from lib.training.callback_trainer import CallbackTrainerSavePredictions

if TYPE_CHECKING:
    from allennlp.training.callback_trainer import (
        CallbackTrainer,
    )  # pylint:disable=unused-import

logger = logging.getLogger(__name__)


@Callback.register("validate_save_predictions")
class ValidateSavePredictions(Validate):
    def __init__(
        self,
        validation_data: Iterable[Instance],
        validation_iterator: DataIterator,
        serialization_dir: str = None,
    ):
        super().__init__(validation_data, validation_iterator)
        self._predictions_output = []
        self._serialization_dir = serialization_dir

    @handle_event(Events.VALIDATE)
    def validate(self, trainer: "CallbackTrainerSavePredictions"):
        # If the trainer has MovingAverage objects, use their weights for validation.
        for moving_average in self.moving_averages:
            moving_average.assign_average_value()

        with torch.no_grad():
            # We have a validation set, so compute all the metrics on it.
            logger.info("Validating")

            trainer.model.eval()

            num_gpus = len(trainer._cuda_devices)  # pylint: disable=protected-access

            raw_val_generator = self.iterator(
                self.instances, num_epochs=1, shuffle=False
            )
            val_generator = lazy_groups_of(raw_val_generator, num_gpus)
            num_validation_batches = math.ceil(
                self.iterator.get_num_batches(self.instances) / num_gpus
            )
            val_generator_tqdm = Tqdm.tqdm(val_generator, total=num_validation_batches)

            batches_this_epoch = 0
            val_loss = 0
            self._predictions_output = []

            for batch_group in val_generator_tqdm:

                loss, outputs = trainer.batch_loss(
                    batch_group, for_training=False, return_all_outputs=True
                )
                if loss is not None:
                    # You shouldn't necessarily have to compute a loss for validation, so we allow for
                    # `loss` to be None.  We need to be careful, though - `batches_this_epoch` is
                    # currently only used as the divisor for the loss function, so we can safely only
                    # count those batches for which we actually have a loss.  If this variable ever
                    # gets used for something else, we might need to change things around a bit.
                    batches_this_epoch += 1
                    val_loss += loss.detach().cpu().numpy()

                # Save output of model
                if len(batch_group) > 1:
                    raise ConfigurationError(
                        "Multiple GPUs are not supported for predict output file"
                    )

                batch_size = trainer.iterator._batch_size

                instance_separated_output: List[Dict[str, List]] = [
                    {} for _ in range(batch_size)
                ]
                for name, output in list(outputs.items()):
                    if isinstance(output, torch.Tensor):
                        # NOTE(markn): This is a hack because 0-dim pytorch tensors are not iterable.
                        # This occurs with batch size 1, because we still want to include the loss in that case.
                        if output.dim() == 0:
                            output = output.unsqueeze(0)

                        if output.size(0) != batch_size:
                            # self._maybe_warn_for_unseparable_batches(name)
                            continue
                        output = output.detach().cpu().numpy()
                    elif len(output) != batch_size:
                        # self._maybe_warn_for_unseparable_batches(name)
                        continue
                    for instance_output, batch_element in zip(
                        instance_separated_output, output
                    ):
                        instance_output[name] = batch_element
                for row in instance_separated_output:
                    self._predictions_output.append(row)

                # Update the description with the latest metrics
                val_metrics = training_util.get_metrics(
                    trainer.model, val_loss, batches_this_epoch
                )
                description = training_util.description_from_metrics(val_metrics)
                val_generator_tqdm.set_description(description, refresh=False)

            trainer.val_metrics = training_util.get_metrics(
                trainer.model, val_loss, batches_this_epoch, reset=True
            )

        # If the trainer has a moving average, restore
        for moving_average in self.moving_averages:
            moving_average.restore()

    @handle_event(Events.EPOCH_END, priority=101)
    def end_of_epoch(self, trainer: "CallbackTrainer"):
        training_states = {}

        # Get attributes from callbacks
        for callback in trainer.handler.callbacks():
            training_states.update(callback.get_training_state())

        is_best_so_far = training_states.pop("is_best_so_far", True)

        if is_best_so_far:
            if self._serialization_dir is None:
                raise ConfigurationError(
                    "serialization_dir is None. Please pass a serialization_dir param"
                )
            path = os.path.join(self._serialization_dir, f"predict_best.json.gz")
            with gzip.open(path, "wt") as pred_file:
                logger.info(f"Saving predictions output file at {path}")
                for row in self._predictions_output:
                    print(json.dumps(sanitize(row)), file=pred_file)
