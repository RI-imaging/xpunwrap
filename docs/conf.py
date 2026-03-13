from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

import sys  # noqa: E402

sys.path.insert(0, str(project_root))
sys.path.append(str(Path(__file__).resolve().parent / "extensions"))

project = "unwrap_phase_gpu"
author = "unwrap_phase_gpu contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "fancy_include",
    "myst_parser",
]

autosummary_generate = True
autodoc_mock_imports = ["cupy"]

templates_path = ["_templates"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
fancy_include_path = "../examples"

html_theme = "sphinx_rtd_theme"
