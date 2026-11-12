"""Include single scripts with doc string, code, and image

Use case
--------
There is an "examples" directory in the root of a repository,
e.g. 'include_doc_code_img_path = "../examples"' in conf.py
(default). An example is a file ("an_example.py") that consists
of a doc string at the beginning of the file, the example code,
and, optionally, an image file (png, jpg) ("an_example.png").


Configuration
-------------
In conf.py, set the parameter

   fancy_include_path = "../examples"

to wherever the included files reside.


Usage
-----
The directive

   .. fancy_include:: an_example.py

will display the doc string formatted with the first line as a
heading, a code block with line numbers, and the image file.
"""
import ast
import os
import pathlib

from docutils.statemachine import ViewList
from docutils.parsers.rst import Directive
from sphinx.util.nodes import nested_parse_with_titles
from docutils import nodes


class IncludeDirective(Directive):
    required_arguments = 1
    optional_arguments = 0

    def run(self):
        env = self.state.document.settings.env
        conf_dir = pathlib.Path(env.app.confdir)
        path = pathlib.Path(env.config.fancy_include_path)
        if not path.is_absolute():
            path = (conf_dir / path).resolve()
        full_path = path / self.arguments[0]

        # Directory of the rst document containing the directive. Image and
        # download paths below are given relative to this directory (no
        # leading "/"), which docutils/Sphinx resolve the same way on every
        # OS. A leading "/" instead means "relative to the source root",
        # which is wrong here since `fancy_include_path` points *outside*
        # the source root (e.g. "../examples") - and using an OS-absolute
        # path directly only happened to "work" on Windows (where it
        # doesn't start with "/" so never triggers that source-root
        # resolution) while silently breaking on Linux build hosts like
        # ReadTheDocs (where it does start with "/").
        doc_dir = pathlib.Path(env.doc2path(env.docname)).parent

        text = full_path.read_text(encoding="utf-8")

        # add reference
        name = full_path.stem
        rst = [".. _example_{}:".format(name),
               "",
               ]

        # Add module docstring and keep full remaining code.
        # Using ast avoids truncation when the file contains additional
        # function/class docstrings after the module-level one.
        module = ast.parse(text)
        module_doc = ast.get_docstring(module, clean=False) or ""
        if module_doc:
            doc = module_doc.splitlines()
            if doc:
                doc.insert(1, "~" * len(doc[0]))  # make title heading
            doc_end_line = module.body[0].end_lineno or 0
            code = text.splitlines()[doc_end_line:]
        else:
            doc = [full_path.stem]
            doc.insert(1, "~" * len(doc[0]))
            code = text.splitlines()

        for line in doc:
            rst.append(line)
        # Ensure subsequent directives are parsed as directives and not as
        # continuation text of the previous paragraph.
        rst.append("")

        # image
        for ext in [".png", ".jpg"]:
            image_path = full_path.with_suffix(ext)
            if image_path.exists():
                break
        else:
            image_path = ""
        if image_path:
            image_rel = pathlib.PurePosixPath(
                os.path.relpath(image_path, doc_dir))
            rst.append(".. figure:: {}".format(image_rel.as_posix()))
            rst.append("")

        # download file
        full_rel = pathlib.PurePosixPath(os.path.relpath(full_path, doc_dir))
        rst.append(":download:`{} <{}>`".format(
            full_path.name, full_rel.as_posix()))

        # code
        rst.append("")
        rst.append(".. code-block:: python")
        rst.append("   :linenos:")
        rst.append("")
        for line in code:
            rst.append("   {}".format(line))
        rst.append("")

        vl = ViewList(rst, "fakefile.rst")
        # Create a node.
        node = nodes.section()
        node.document = self.state.document
        # Parse the rst.
        nested_parse_with_titles(self.state, vl, node)
        return node.children


def setup(app):
    app.add_config_value('fancy_include_path', "../examples", 'html')

    app.add_directive('fancy_include', IncludeDirective)

    return {'version': '0.1'}   # identifies the version of our extension
