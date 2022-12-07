# Copyright 2022 The jax_tpu_embedding Authors.
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

"""Pytype Utils."""

from typing import Any, Dict, List, Tuple, TypeVar, Union, Iterable

import tensorflow as tf
# pylint: disable=g-direct-tensorflow-import
from tensorflow..core.protobuf.tpu import tpu_embedding_configuration_pb2
# pylint: enable=g-direct-tensorflow-import


T = TypeVar("T")
Nested = Union[T, Tuple[Any, ...], List[Any], Dict[str, Any], Iterable]

TensorType = Union[tf.Tensor, tf.SparseTensor]
NestedTfTensor = Nested[tf.Tensor]

# TPUEmbedding related Types
TableConfig = tf.tpu.experimental.embedding.TableConfig
FeatureConfig = tf.tpu.experimental.embedding.FeatureConfig
NestedFeatureConfig = Nested[FeatureConfig]
TPUEmbeddingConfigurationProto = tpu_embedding_configuration_pb2.TPUEmbeddingConfiguration


