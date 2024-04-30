# Copyright 2024 The jax_tpu_embedding Authors.
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

from typing import Any, Dict, List, Mapping, Tuple, TypeVar, Union, Iterable

import jax
from jax_tpu_embedding import checkpoint
import tensorflow as tf
# pylint: disable=g-direct-tensorflow-import
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.protobuf.tpu import tpu_embedding_configuration_pb2
from tensorflow.python.tpu import tpu_embedding_v2_utils
# pylint: enable=g-direct-tensorflow-import


T = TypeVar('T')
Nested = Union[T, Tuple[Any, ...], List[Any], Dict[str, Any], Iterable]
NestedStruct = Union[Dict[str, Dict[str, T]], Mapping[str, Mapping[str, T]]]

TensorType = Union[tf.Tensor, tf.SparseTensor]
TensorProto = tensor_pb2.TensorProto
NestedTfTensor = Nested[tf.Tensor]
NestedJaxArray = Nested[jax.Array]

# TPUEmbedding related Types
TableConfig = tf.tpu.experimental.embedding.TableConfig
FeatureConfig = tf.tpu.experimental.embedding.FeatureConfig
NestedFeatureConfig = Nested[FeatureConfig]
TPUEmbeddingConfigurationProto = (
    tpu_embedding_configuration_pb2.TPUEmbeddingConfiguration
)
TPUEmbeddingOptimizer = tpu_embedding_v2_utils._Optimizer  # pylint: disable=protected-access

# TPUEmbedding checkpoints
GlobalHostArray = checkpoint.GlobalHostArray
RestoreArgs = checkpoint.GlobalHostArrayRestoreArgs

# Special field names.
EMBED_PLACEMENT = 'host'
NON_EMBED_PLACEMENT = 'device'
EMBED_ACTV_KEY = 'embedding_actvs'
