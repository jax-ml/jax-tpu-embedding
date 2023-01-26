"""Implements custom rules for jax_tpu_embedding."""

# Placeholder to use until bazel supports pytype_*.
def pytype_strict_library(name, **kwargs):
    native.py_library(name = name, **kwargs)

def pytype_library(name, **kwargs):
    native.py_library(name = name, **kwargs)

def pytype_binary(name, **kwargs):
    native.py_binary(name = name, **kwargs)

def pytype_strict_binary(name, **kwargs):
    native.py_binary(name = name, **kwargs)

# Placeholder to use until bazel supports py_strict_test.
def py_strict_test(name, **kwargs):
    native.py_test(name = name, **kwargs)
