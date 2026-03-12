from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

import sys  # noqa: E402

sys.path.insert(0, str(project_root))

project = "unwrap_phase_gpu"
author = "unwrap_phase_gpu contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autodoc_mock_imports = ["cupy"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
