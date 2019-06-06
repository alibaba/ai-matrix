# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utils for creating PerfZero benchmarks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
from absl.testing import flagsaver
import tensorflow as tf  # pylint: disable=g-bad-import-order

FLAGS = flags.FLAGS


class PerfZeroBenchmark(tf.test.Benchmark):
  """Common methods used in PerfZero Benchmarks.

     Handles the resetting of flags between tests, loading of default_flags,
     overriding of defaults.  PerfZero (OSS) runs each test in a separate
     process reducing some need to reset the flags.
  """
  local_flags = None

  def __init__(self, output_dir=None, default_flags=None, flag_methods=None):
    """Initialize class.

    Args:
      output_dir: Base directory to store all output for the test.
      default_flags:
      flag_methods:
    """
    if not output_dir:
      output_dir = '/tmp'
    self.output_dir = output_dir
    self.default_flags = default_flags or {}
    self.flag_methods = flag_methods or {}

  def _get_model_dir(self, folder_name):
    """Returns directory to store info, e.g. saved model and event log."""
    return os.path.join(self.output_dir, folder_name)

  def _setup(self):
    """Sets up and resets flags before each test."""
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)
    if PerfZeroBenchmark.local_flags is None:
      for flag_method in self.flag_methods:
        flag_method()
      # Loads flags to get defaults to then override. List cannot be empty.
      flags.FLAGS(['foo'])
      # Overrides flag values with defaults for the class of tests.
      for k, v in self.default_flags.items():
        setattr(FLAGS, k, v)
      saved_flag_values = flagsaver.save_flag_values()
      PerfZeroBenchmark.local_flags = saved_flag_values
    else:
      flagsaver.restore_flag_values(PerfZeroBenchmark.local_flags)
