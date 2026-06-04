from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from fungi_rag.models import load_brief_yaml
from fungi_rag.workflow import FungiWorkflow


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a fungi research workflow headlessly.")
    parser.add_argument("brief", help="Path to a YAML research brief.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    brief = load_brief_yaml(Path(args.brief))
    state, paths = FungiWorkflow().run_research(brief)
    print(f"Run {state.run_id}: {state.status}")
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
