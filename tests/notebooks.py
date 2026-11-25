"""Load model-development notebooks in tests, since notebooks can't be imported."""
import json
from pathlib import Path

MODEL_DEVELOPMENT = Path(__file__).resolve().parents[1] / "src" / "model_development"


def load_notebook(path, namespace=None):
    """Run a notebook's code cells into a namespace dict and return it.
    `%run other.ipynb` lines load that notebook first, relative to this one (as Jupyter does when the notebook is
    opened from its own folder)."""
    path = Path(path)
    namespace = {"__name__": "notebook"} if namespace is None else namespace
    for cell in json.loads(path.read_text())["cells"]:
        if cell["cell_type"] != "code":
            continue
        code = []
        for line in "".join(cell["source"]).splitlines():
            if line.startswith("%run "):
                load_notebook(path.parent / line[5:].strip(), namespace)
            else:
                code.append(line)
        exec(compile("\n".join(code), f"{path.name}", "exec"), namespace)
    return namespace


def training(name):
    return load_notebook(MODEL_DEVELOPMENT / "training" / f"{name}.ipynb")


def val(name):
    return load_notebook(MODEL_DEVELOPMENT / "val" / f"{name}.ipynb")
