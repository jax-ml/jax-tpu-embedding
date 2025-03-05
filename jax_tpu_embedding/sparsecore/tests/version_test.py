# Copyright 2024 The JAX SC Authors.
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
"""Test version string generation."""

from absl.testing import absltest
from jax_tpu_embedding import sparsecore
from jax_tpu_embedding.sparsecore import version


class VersionTest(absltest.TestCase):

  def test_version_string(self):
    self.assertEqual(sparsecore.__version__, version.__version__)
    self.assertTrue(version.__version__.startswith(version._base_version))
    self.assertTrue(version.__version__.endswith(version._version_suffix))


if __name__ == "__main__":
  absltest.main()
