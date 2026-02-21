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
"""Configuration file for the Sphinx documentation builder."""

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import pathlib
import shutil
import sys

Path = pathlib.Path

# Skip building package for docs instead add py source files to sys path.
sys.path.insert(0, str(Path('../../..').resolve()))
shutil.copy(
    '../version.py.in',
    '../version.py',
)

project = 'jax-tpu-embedding'
# The copyright variable is used for building the docs but it leads to a
# redefined-builtin error. We ignore the error here as a workaround.
copyright = '2026, JAX TPU Embedding Team'  # pylint: disable=redefined-builtin
author = 'JAX TPU Embedding Team'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_nb',
    'sphinx_design',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.graphviz',
]

templates_path = ['_templates']
exclude_patterns = ['notebooks/*.md']

intersphinx_mapping = {
    'jax': ('https://jax.readthedocs.io/en/latest/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_title = 'JAX TPU Embedding'
html_theme_options = {
    'show_navbar_depth': 1,
    'show_toc_level': 2,
    'repository_url': 'https://github.com/jax-ml/jax-tpu-embedding',
    'use_issues_button': True,
    'use_repository_button': True,
    'navigation_with_keys': True,
}

autosummary_generate = True
autodoc_typehints = 'description'
add_module_names = False

autodoc_mock_imports = [
    'jax_tpu_embedding.sparsecore.lib.core.pybind_input_preprocessing',
    'jax_tpu_embedding.sparsecore.lib.proto.embedding_spec_pb2',
    'einops',
]

# -- Myst configurations -------------------------------------------------
nb_execution_mode = 'off'
source_suffix = ['.rst', '.ipynb', '.md']
