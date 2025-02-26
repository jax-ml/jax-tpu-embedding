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
"""Build script for pip package."""

from collections.abc import Sequence
import subprocess
import sys

from absl import app


def main(argv: Sequence[str]) -> None:
  if len(argv) < 2:
    raise app.UsageError(f"Usage: \n  {argv[0]} <output path>")

  output_path = argv[1]
  print("Building wheels in", output_path)

  python_command = sys.executable
  try:
    process = subprocess.run(
        [python_command, "-m", "build", "--outdir", output_path],
        check=True,
        capture_output=True,
        text=True,
    )
    print("Build output:")
    print(process.stdout)
  except subprocess.CalledProcessError as e:
    print(f"Error executing build: {e}")
    print(e.stderr)


if __name__ == "__main__":
  app.run(main)
