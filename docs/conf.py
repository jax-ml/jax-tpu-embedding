# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sys
from pathlib import Path
import shutil

# Skip building package for docs
sys.path.insert(0, str(Path('..').resolve()))
shutil.copy(
  '../jax_tpu_embedding/sparsecore/version.py.in',
  '../jax_tpu_embedding/sparsecore/version.py',
)

project = 'jax-tpu-embedding'
copyright = '2026, JAX TPU Embedding Team'
author = 'JAX TPU Embedding Team'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx_design',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
]

templates_path = ['_templates']
exclude_patterns = []

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
