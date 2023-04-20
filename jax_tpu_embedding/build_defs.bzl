"""Starlark macros for Jax Embedding API users."""

load("//jax_tpu_embedding:jte.bzl", "pytype_binary", "pytype_strict_binary")

def _export_sources_impl(ctx):
    files = []
    for dep in ctx.attr.deps:
        files += dep[DefaultInfo].files.to_list()

    srcs = depset(direct = files)
    return [
        DefaultInfo(files = srcs),
    ]

# This rule defines a target which contains the list of source files
# (`srcs`) from a list of Python library dependencies in `deps`.
#
# For example, if we have:
#   py_library(name = "my_lib", srcs = ["my_lib.py"], ...)
# , then
#   export_sources(name = "lib_files", deps = [":my_lib"])
# defines a target ":lib_files" which is [":my_lib.py"].
export_sources = rule(
    implementation = _export_sources_impl,
    attrs = {
        "deps": attr.label_list(
            providers = [PyInfo],
            allow_empty = True,
            mandatory = True,
            cfg = "exec",
        ),
    },
)

def jte_targets(
        experiments = None,
        extra_deps = None,
        prefix_name = "",
        name = "",
        main_src = "//third_party/py/jax_tpu_embedding:tpu_embedding.py"):
    """Macro to define a collection of Jax TPU Embedding targets with custom dependencies.

    It currently defines the following targets:

    ":main", a Python binary that can be passed to the xm launcher to run
        the experiments.

    Args:
      experiments: a list of py_library targets that defines and registers all
          experiments for this collection. Experiments should be registered
          when the `srcs` files are imported.
      extra_deps: a list of extra dependencies not already included.
      prefix_name: string, a common prefix for the generated targets.
          An underscore is added, e.g. if prefix_name="test", the defined
          main target is ":test_main".
      name: unused.
      main_src: The src file for the ":main" target created.
    """
    if not experiments:
        fail("jte_targets() expects a non-empty list of deps that defines " +
             "and registers experiments.")
    if name:
        fail("name is not used and has no effect. Specify prefix_name instead.")

    exp_sources = ("_exp_sources" if not prefix_name else "_%s_exp_sources" % prefix_name)
    export_sources(
        name = exp_sources,
        deps = experiments,
    )
    extra_deps = experiments + (extra_deps or [])

    main_name = "tpu_embedding"
    main_name = main_name if not prefix_name else "%s_%s" % (prefix_name, main_name)
    export_binary(
        name = main_name,
        main = main_src,
        py_binary_rule = pytype_binary,
        deps = [
            "//third_party/py/jax_tpu_embedding:tpu_embedding",
        ] + extra_deps,
        exp_sources = exp_sources,
        paropts = ["--compress"],
    )

def export_binary(
        name,
        main,
        deps,
        py_binary_rule,
        exp_sources,
        **kwargs):
    """Define an existing `py_binary()` at the current package.

    Args:
      name: name of the generated rule.
      main: Binary src.
      deps: Dependencies required by binary src.
      py_binary_rule: the Blaze rule to use to create the final binary.
      exp_sources: target of experiment source files.
      **kwargs: all remaining arguments are passed through.
    """
    main_copied = "%s.py" % name
    _copy_src(output_name = main_copied, source_target = main, exp_sources = exp_sources)

    # Main script.
    py_binary_rule(
        name = name,
        python_version = "PY3",
        main = main_copied,
        srcs = [main_copied],
        deps = deps,
        **kwargs
    )

def _copy_src(output_name, source_target, exp_sources):
    # To avoid build warning when using `srcs` on a `py_binary()` outside the
    # current package, copy the file locally with a new rule.
    # We also prepend the source file with imports that registers all
    # experiments.
    native.genrule(
        name = output_name + ".copy",
        outs = [output_name],
        srcs = [source_target, exp_sources],
        cmd = """cat <<EOF > $@ && cat $(location %s) >> $@
# Auto-generated code to import and register all experiments.
import importlib
import_str = '$(locations %s)'
for d in import_str.split(' '):
  assert d.endswith('.py'), d
  d = d.replace('/', '.')[:-len('.py')]
  # internal build_defs.bzl imports code
  importlib.import_module(d)
# End of auto-generated code.

EOF
        """ % (source_target, exp_sources),
    )
