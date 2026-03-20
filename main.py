import shutil
from pathlib import Path

PIPELINE_FILES = [
    "initialization.py",
    "eda.py",
    "preprocessing.py",
    "test_split.py",
    "train_models.py",
    "validation.py",
    "summary.py",
]

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"


def clear_output_dir():
    if not OUTPUT_DIR.exists():
        return

    for path in OUTPUT_DIR.iterdir():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def run_pipeline():
    clear_output_dir()
    OUTPUT_DIR.mkdir(exist_ok=True)

    shared_globals = {
        "__builtins__": __builtins__,
        "__name__": "__main__",
        "__package__": None,
    }

    for filename in PIPELINE_FILES:
        file_path = BASE_DIR / filename
        print(f"\n{'=' * 80}\nRunning {filename}\n{'=' * 80}")
        shared_globals["__file__"] = str(file_path)
        code = compile(file_path.read_text(encoding="utf-8"), str(file_path), "exec")
        exec(code, shared_globals)

    return shared_globals


if __name__ == "__main__":
    run_pipeline()
