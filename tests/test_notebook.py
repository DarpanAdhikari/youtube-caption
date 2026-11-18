import ast
import json
from pathlib import Path

import pytest

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"


@pytest.mark.parametrize(
    "path",
    sorted(NOTEBOOK_DIR.glob("*.ipynb")),
    ids=lambda path: path.name,
)
def test_notebook_code_cells_parse(path: Path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    assert code_cells
    for cell in code_cells:
        ast.parse("".join(cell["source"]))
