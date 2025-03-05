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
"""Generate a file using a template."""

load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")

def _get_flag_substitutions(flag_substitutions):
    """Extracts flag values."""
    substitutions = {}
    for key, label in flag_substitutions.items():
        substitutions[key] = label[BuildSettingInfo].value
    return substitutions

def _create_substitution_map(string_substitutions):
    """Replaces {key: value} with {${key}: value}"""
    substitutions = {}
    for key, value in string_substitutions.items():
        key_var = "${" + key + "}"
        substitutions[key_var] = value
    return substitutions

def configure_file(
        name,
        template,
        output,
        substitutions = {},
        flag_substitutions = {}):
    """Generates a file using a template.

    For every entry in the substitutions maps, replaces `${variable}` with `value`.

    Args:
        name: The name of the rule.
        template: The template file in which to perform the substitutions.
        output: The output file.
        substitutions: A map of string substitutions {variable: value}.
        flag_substitutions: A map of variable to bazel string_flag substitutions. \
            Replacement values are extracted from the flag.

    Returns:
        A rule that generates the output file.
    """
    _configure_file(
        name = name,
        template = template,
        output = output,
        substitutions = substitutions,
        flag_substitutions = flag_substitutions,
    )

def _configure_file_impl(ctx):
    substitutions = _create_substitution_map(ctx.attr.substitutions | _get_flag_substitutions(ctx.attr.flag_substitutions))
    ctx.actions.expand_template(
        template = ctx.file.template,
        output = ctx.outputs.output,
        substitutions = substitutions,
    )

_configure_file = rule(
    implementation = _configure_file_impl,
    attrs = {
        "template": attr.label(
            mandatory = True,
            allow_single_file = True,
        ),
        "substitutions": attr.string_dict(),
        "flag_substitutions": attr.string_keyed_label_dict(),
        "output": attr.output(mandatory = True),
    },
)
