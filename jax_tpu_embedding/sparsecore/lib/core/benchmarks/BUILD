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
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//jax_tpu_embedding/sparsecore:jax_tpu_embedding.bzl", "CORE_USERS")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = CORE_USERS,
)

cc_test(
    name = "input_preprocessing_benchmark",
    srcs = ["input_preprocessing_benchmark.cc"],
    deps = [
        "//jax_tpu_embedding/sparsecore/lib/core:abstract_input_batch",
        "//jax_tpu_embedding/sparsecore/lib/core:input_preprocessing_core",
        "//jax_tpu_embedding/sparsecore/lib/core:input_preprocessing_util",
        "//jax_tpu_embedding/sparsecore/lib/core:partitioned_coo_tensors",
        "//jax_tpu_embedding/sparsecore/lib/core:ragged_tensor_input_batch",
        "//jax_tpu_embedding/sparsecore/lib/core:sort_and_group_coo_tensors_impl",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/random:distributions",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_benchmark//:benchmark_main",
        "@eigen_archive//:eigen3",
    ],
)
