load("//devtools/python/blaze:pytype.bzl", "pytype_strict_contrib_test", "pytype_strict_library")

package(
    default_testonly = True,
)

pytype_strict_library(
    name = "test_utils",
    testonly = False,
    srcs = ["test_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/jax",
        "//third_party/py/jax_tpu_embedding:pytype_utils",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_contrib_test(
    name = "input_utils_test",
    srcs = ["input_utils_test.py"],
    python_version = "PY3",
    deps = [
        ":test_utils",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/jax",
        "//third_party/py/jax:experimental",
        "//third_party/py/jax_tpu_embedding:input_utils",
        "//third_party/py/tensorflow",
    ],
)
