from __future__ import annotations

import html
from pathlib import Path

from fungi_rag.citation import format_references
from fungi_rag.models import AgentState, EvidencePacket, GenerationResult, RagTrace
from fungi_rag.utils import ensure_dir, write_json


class Exporter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = ensure_dir(output_dir)

    def export_run(
        self,
        *,
        state: AgentState,
        evidence: EvidencePacket,
        generation_results: list[GenerationResult],
        traces: list[RagTrace],
    ) -> dict[str, str]:
        report = self._report_markdown(state, evidence, generation_results)
        report_path = self.output_dir / "report.md"
        html_path = self.output_dir / "report.html"
        report_path.write_text(report, encoding="utf-8")
        html_path.write_text(markdownish_to_html(report), encoding="utf-8")
        sources_path = write_json(
            self.output_dir / "sources.json",
            [item.model_dump(mode="json") for item in evidence.items],
        )
        citations_path = write_json(
            self.output_dir / "citations.json",
            [item.citation().model_dump(mode="json") for item in evidence.items],
        )
        state_path = write_json(self.output_dir / "state.json", state.model_dump(mode="json"))
        trace_path = write_json(
            self.output_dir / "rag_trace.json",
            [trace.model_dump(mode="json") for trace in traces],
        )
        return {
            "report": str(report_path),
            "html": str(html_path),
            "sources": str(sources_path),
            "citations": str(citations_path),
            "state": str(state_path),
            "rag_trace": str(trace_path),
        }

    def _report_markdown(
        self,
        state: AgentState,
        evidence: EvidencePacket,
        generation_results: list[GenerationResult],
    ) -> str:
        parts = [f"# {state.brief.title if state.brief else 'Fungi RAG Report'}"]
        for result in generation_results:
            parts.append(f"\n## {result.step.title()}\n")
            if result.text:
                parts.append(result.text)
            else:
                parts.append(
                    "Generation is pending. Complete the Codex response file listed in "
                    f"`{result.response_path}` and rerun the workflow."
                )
                if result.validation_errors:
                    parts.append("\nValidation notes:\n" + "\n".join(f"- {e}" for e in result.validation_errors))
        parts.append("\n" + format_references(evidence))
        return "\n\n".join(parts)


def markdownish_to_html(markdown: str) -> str:
    lines = []
    for raw in markdown.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("# "):
            lines.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            lines.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("- "):
            lines.append(f"<li>{html.escape(line[2:])}</li>")
        else:
            lines.append(f"<p>{html.escape(line)}</p>")
    return "<!doctype html><html><body>" + "\n".join(lines) + "</body></html>"
