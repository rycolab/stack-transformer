# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example script to train and evaluate a network."""

from absl import app

import haiku as hk
import jax.numpy as jnp
import numpy as np
import jax
import os

import sys
sys.path.append('../')

from neural_networks_chomsky_hierarchy.training import constants
from neural_networks_chomsky_hierarchy.training import curriculum as curriculum_lib
from neural_networks_chomsky_hierarchy.training import training
from neural_networks_chomsky_hierarchy.training import utils
from neural_networks_chomsky_hierarchy.models import positional_encodings as pos_encs_lib

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

def main(unused_argv) -> None:
  scores = []
  for seed in [0, 1, 2, 3, 4]:
    # Change your hyperparameters here. See constants.py for possible tasks and
    # architectures.
    pos = pos_encs_lib.PositionalEncodings.NONE
    lr = 1e-4

    batch_size = 32
    sequence_length = 40
    max_range_test_length = 100
    task = 'reverse_string'
    architecture = 'transformer_encoder'
    stack = True

    is_autoregressive = False
    causal_masking = False

    if architecture == 'transformer_decoder':
      causal_masking = True
      is_autoregressive = True

    max_time = 256
    architecture_params = {'positional_encodings': pos, 'positional_encodings_params': pos_encs_lib.SinCosParams(max_time=max_time), 
                            'stack': stack, 'feedforward': True, 'attention': True, 'num_layers': 5, 'causal_masking': causal_masking}

    # Create the task.
    curriculum = curriculum_lib.UniformCurriculum(
        values=list(range(1, sequence_length + 1)))
    task = constants.TASK_BUILDERS[task](2)

    # Create the model.

    computation_steps_mult = 0
    single_output = task.output_length(10) == 1

    if single_output:
      is_autoregressive = False

    model = constants.MODEL_BUILDERS[architecture](
        output_size=task.output_size,
        return_all_outputs=True,
        **architecture_params)
    if is_autoregressive:
      if architecture != 'transformer':
        model = utils.make_model_with_targets_as_input(
            model, computation_steps_mult
        )
      model = utils.add_sampling_to_autoregressive_model(model, single_output)
    else:
      model = utils.make_model_with_empty_targets(
          model, task, computation_steps_mult, single_output
      )
    model = hk.transform(model)

    # Create the loss and accuracy based on the pointwise ones.
    def loss_fn(output, target):
      loss = jnp.mean(jnp.sum(task.pointwise_loss_fn(output, target), axis=-1))
      return loss, {}

    def accuracy_fn(output, target):
      mask = task.accuracy_mask(target)
      return jnp.sum(mask * task.accuracy_fn(output, target)) / jnp.sum(mask)

    # Create the final training parameters.
    training_params = training.ClassicTrainingParams(
        seed=seed,
        model_init_seed=seed,
        training_steps=1_00_000,
        log_frequency=100,
        length_curriculum=curriculum,
        batch_size=batch_size,
        task=task,
        model=model,
        loss_fn=loss_fn,
        learning_rate=lr,
        accuracy_fn=accuracy_fn,
        compute_full_range_test=True,
        max_range_test_length=max_range_test_length,
        range_test_total_batch_size=512,
        range_test_sub_batch_size=32,
        is_autoregressive=is_autoregressive)

    training_worker = training.TrainingWorker(training_params, use_tqdm=True)
    _, eval_results, _ = training_worker.run()

    # Gather results and print final score.
    accuracies = [r['accuracy'] for r in eval_results]
    score = np.mean(accuracies[sequence_length + 1:])
    print(f'seed: {seed}, learning rate: {lr}')
    print(f'Network score: {score}')
    scores.append(score)
    jax.clear_caches()
    jax.clear_backends()
  print(f'Maximum score: {max(scores)}')
  print(f'Mean score: {np.mean(scores)}')
  print(f'Std score: {np.std(scores)}')

if __name__ == '__main__':
  app.run(main)
