from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import yaml

from fungi_rag.config import get_settings
from fungi_rag.ingest import DocumentIngestor
from fungi_rag.models import ResearchBrief
from fungi_rag.sources import SourceDownloader, load_manifest
from fungi_rag.workflow import FungiWorkflow


class MissingGradioApp:
    def launch(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("gradio is not installed. Run `python -m pip install -e .` first.")


def build_app():
    try:
        import gradio as gr
    except ImportError:
        return MissingGradioApp()

    settings = get_settings()

    with gr.Blocks(title="Fungi RAG Learning System") as demo:
        gr.Markdown("# Fungi RAG Learning System")
        with gr.Tab("Brief"):
            brief_text = gr.Code(
                label="YAML research brief",
                language="yaml",
                value=Path("examples/fungi_research_brief.yaml").read_text(encoding="utf-8")
                if Path("examples/fungi_research_brief.yaml").exists()
                else "",
            )
            validate_button = gr.Button("Validate Brief")
            brief_status = gr.JSON(label="Validation")
            validate_button.click(validate_brief_ui, inputs=brief_text, outputs=brief_status)

        with gr.Tab("Corpus"):
            manifest_path = gr.Textbox(label="Source manifest", value="examples/source_manifest.yaml")
            download_button = gr.Button("Download Seed Academic Corpus")
            download_status = gr.JSON(label="Download status")
            ingest_path = gr.Textbox(label="Path to ingest", value=str(settings.source_raw_dir))
            ingest_button = gr.Button("Ingest Path")
            ingest_status = gr.JSON(label="Ingestion status")
            download_button.click(download_sources_ui, inputs=manifest_path, outputs=download_status)
            ingest_button.click(ingest_ui, inputs=ingest_path, outputs=ingest_status)

        with gr.Tab("Research Run"):
            run_button = gr.Button("Create Codex Outline/Draft Packets")
            run_status = gr.JSON(label="Run status")
            run_button.click(run_research_ui, inputs=brief_text, outputs=run_status)

        with gr.Tab("Ask"):
            question = gr.Textbox(label="Learner question")
            ask_button = gr.Button("Retrieve Evidence and Create Codex Packet")
            answer = gr.JSON(label="Result")
            ask_button.click(ask_ui, inputs=question, outputs=answer)

        with gr.Tab("Evaluation"):
            evaluate_button = gr.Button("Run Evaluation")
            eval_status = gr.JSON(label="Evaluation")
            evaluate_button.click(evaluate_ui, outputs=eval_status)

    return demo


def validate_brief_ui(text: str) -> dict[str, Any]:
    try:
        brief = ResearchBrief.model_validate(yaml.safe_load(text))
        return {"ok": True, "brief": brief.model_dump(mode="json")}
    except Exception as exc:  # noqa: BLE001 - UI should return validation detail.
        return {"ok": False, "error": str(exc)}


def download_sources_ui(manifest_path: str) -> dict[str, Any]:
    manifest = load_manifest(Path(manifest_path))
    rows = SourceDownloader().download_manifest(manifest)
    return {"count": len(rows), "rows": rows}


def ingest_ui(path: str) -> dict[str, Any]:
    chunks = DocumentIngestor().ingest_path(Path(path))
    return {"chunks": len(chunks), "path": path}


def run_research_ui(text: str) -> dict[str, Any]:
    try:
        brief = ResearchBrief.model_validate(yaml.safe_load(text))
        state, paths = FungiWorkflow().run_research(brief)
        return {"state": state.model_dump(mode="json"), "paths": paths}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def ask_ui(question: str) -> dict[str, Any]:
    return FungiWorkflow().ask(question)


def evaluate_ui() -> dict[str, Any]:
    from fungi_rag.evaluate import run_evaluation

    return run_evaluation()


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Launch the Fungi RAG Gradio app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(list(argv) if argv is not None else None)
    app = build_app()
    app.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
