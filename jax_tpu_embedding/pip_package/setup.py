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

"""Setup.py file for jax_tpu_embedding."""

from setuptools import find_packages
from setuptools import setup

install_requires = [
    'tensorflow',
    'keras_applications',
    'keras_preprocessing',
    'jax',
]

setup(
    name='jax_tpu_embedding',
    version='0.1.0',
    description=('TPU Embedding support for Jax users'),
    author='Jax TPU Embedding Team',
    author_email='jax-tpu-embedding-dev@google.com',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=install_requires,
    url='https://github.com/jax-ml/jax-tpu-embedding',
    license='Apache-2.0',
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    zip_safe=False,
)
