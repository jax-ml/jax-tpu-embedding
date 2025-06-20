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
"""Build script for pip wheel."""

from collections.abc import Sequence
import os
import platform
import re
import subprocess
import sys
from absl import app
from absl import flags
from absl import logging


_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir', None, 'Output directory for generated wheels.', short_name='o'
)


def _extract_wheel_info(filename: str) -> dict[str, str]:
  """Extracts version and tag information from the wheel name.

  According to PEP-0427.

  Args:
    filename: wheel filename.

  Returns:
    Dictionary with the filename components: distribution, version, build_tag,
    python_tag, abi_tag, platform_tag.

  Raises:
    RuntimeError: if parsing the filename fails.
  """
  # pylint:disable=line-too-long
  # https://peps.python.org/pep-0427/
  # {distribution}-{version}(-{build tag})?-{python tag}-{abi tag}-{platform tag}.whl
  # pylint:enable=line-too-long
  wheel_info_re = re.compile(
      (
          r'^((?P<distribution>[^\s-]+?)-(?P<version>[^\s-]+?))'
          r'(-(?P<build_tag>\d[^\s-]*))?-(?P<python_tag>[^\s-]+?)'
          r'-(?P<abi_tag>[^\s-]+?)-(?P<platform_tag>\S+)\.whl$'
      ),
  )
  wheel_info = wheel_info_re.match(filename)
  if not wheel_info:
    raise RuntimeError(f'Unable to extract wheel info from {filename}')

  return wheel_info


def run_process(cmd: Sequence[str]):
  """Run a command as a subprocess, checking the return code.

  Args:
    cmd: process string tokens.

  Returns:
    The subprocess.run result.

  Raises:
    RuntimeError: on failure.
  """
  process = subprocess.run(cmd, check=False, capture_output=True, text=True)
  if process.returncode != 0:
    logging.error('Process failed with exit code %s', process.returncode)
    logging.error('Command: %s', ' '.join(cmd))
    logging.error('stdout: %s', process.stdout)
    logging.error('stderr: %s', process.stderr)
    raise RuntimeError(f'Process failed with exit code {process.returncode}')

  return process


def run_build(output_dir: str) -> str:
  """Builds the wheel using the python `build` package.

  Args:
    output_dir: wheel output directory.

  Returns:
    Generated filenames for binary and source distributions, respectively.

  Raises:
    subprocess.CalledProcessError: if the command fails.
    RuntimeError: if we fail to parse the output of the build command.
  """
  logging.info('Building wheels in %s', output_dir)
  process = run_process(
      [sys.executable, '-m', 'build', '--outdir', output_dir],
  )
  stdout = process.stdout
  logging.debug('Build output:\n%s', stdout)

  # Extract wheel information.
  build_info = re.search(
      r'Successfully built (?P<sdist>[^\s]+) and (?P<bdist>[^\s]+)', stdout
  )
  if not build_info:
    raise RuntimeError('Unable to find sdist and bdist in build output.')

  bdist = build_info['bdist']
  sdist = build_info['sdist']
  return bdist, sdist


def run_auditwheel_show(bdist_path: str) -> str:
  """Runs `auditwheel show` on the provided wheel file.

  Args:
    bdist_path: full path of the binary distribution wheel.

  Returns:
    Supported platform name `plat` from auditwheel.

  Raises:
    subprocess.CalledProcessError: if the command fails.
    RuntimeError: if we fail to parse the output of the command.
  """
  logging.info('Running auditwheel show on %s', bdist_path)
  process = run_process(
      [sys.executable, '-m', 'auditwheel', 'show', bdist_path],
  )
  stdout = process.stdout
  logging.debug(stdout)

  # Potentially fix wheel based on compatiability tag.
  auditwheel_info = re.search(
      r'This constrains the platform tag to "(?P<plat>[\w]+?)"', stdout
  )
  if not auditwheel_info:
    raise RuntimeError(
        f'Unable to determine auditwheel platform from output:\n {stdout}'
    )

  auditwheel_plat = auditwheel_info['plat']
  return auditwheel_plat


def run_auditwheel_repair(
    bdist_path: str, platform_tag: str, output_dir: str
) -> str:
  """Runs `auditwheel repair` on the provided wheel file.

  Args:
    bdist_path: full path of the original binary distribution wheel.
    platform_tag: auditwheel platform tag, e.g. manylinux2014_x86_64.
    output_dir: output directory for repaired wheels.

  Returns:
    Full path of the repaired wheel.

  Raises:
    subprocess.CalledProcessError: if the command fails.
    RuntimeError: if we fail to parse the output of the command.
  """
  logging.info(
      'Running auditwheel repair on %s with platform tag %s',
      bdist_path,
      platform_tag,
  )
  process = run_process(
      [
          sys.executable,
          '-m',
          'auditwheel',
          'repair',
          '--plat',
          platform_tag,
          '-w',
          output_dir,
          bdist_path,
      ],
  )
  # Auditwheel repair outputs to stderr.
  stderr = process.stderr
  logging.debug(stderr)

  auditwheel_info = re.search(
      r'Fixed-up wheel written to (?P<wheel>[\S]+)', stderr
  )
  if not auditwheel_info:
    raise RuntimeError(
        f'Unable to determine repaired wheel from auditwheel output:\n {stderr}'
    )

  return auditwheel_info['wheel']


def main(argv: Sequence[str]) -> None:
  del argv

  output_dir = _OUTPUT_DIR.value
  if not output_dir:
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'dist'
    )

  logging.info('Platform: %s, %s', platform.system(), platform.machine())

  # Build wheel.
  bdist, _ = run_build(output_dir)
  wheel_info = _extract_wheel_info(bdist)

  # Run auditwheel to check and repair compatibility.
  bdist_path = os.path.join(output_dir, bdist)
  auditwheel_plat = run_auditwheel_show(bdist_path)

  if auditwheel_plat != wheel_info['platform_tag']:
    repaired_path = run_auditwheel_repair(
        bdist_path, auditwheel_plat, output_dir
    )
    # Remove unrepaired wheel.
    if repaired_path != bdist_path:
      logging.debug('Removing unrepaired wheel: %s', bdist_path)
      os.remove(bdist_path)
      bdist_path = repaired_path

  logging.info('Final wheel: %s', bdist_path)


if __name__ == '__main__':
  app.run(main)
