from __future__ import annotations

import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

from fungi_rag.citation import CitationAuditor
from fungi_rag.models import GenerationRequest, GenerationResult
from fungi_rag.safety import SafetyPolicy
from fungi_rag.utils import atomic_write_text, write_json


class Generator(ABC):
    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        raise NotImplementedError


class CodexBridgeGenerator(Generator):
    def __init__(self) -> None:
        self.auditor = CitationAuditor()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        task_dir = request.output_dir / "codex_tasks"
        task_dir.mkdir(parents=True, exist_ok=True)
        base = task_dir / request.step
        prompt_path = base.with_suffix(".prompt.md")
        evidence_path = base.with_suffix(".evidence.json")
        schema_path = base.with_suffix(".schema.json")
        response_path = base.with_suffix(".response.md")

        atomic_write_text(prompt_path, self._render_prompt(request))
        write_json(evidence_path, request.evidence.model_dump(mode="json"))
        write_json(schema_path, request.response_schema or default_response_schema(request.step))

        if not response_path.exists():
            return GenerationResult(
                status="pending",
                step=request.step,
                prompt_path=str(prompt_path),
                evidence_path=str(evidence_path),
                schema_path=str(schema_path),
                response_path=str(response_path),
                validation_errors=[
                    "Codex response is pending. Complete the response file, then rerun validation."
                ],
            )

        text = response_path.read_text(encoding="utf-8")
        errors = self.validate_text(text, request)
        return GenerationResult(
            status="accepted" if not errors else "invalid",
            step=request.step,
            prompt_path=str(prompt_path),
            evidence_path=str(evidence_path),
            schema_path=str(schema_path),
            response_path=str(response_path),
            text=text,
            validation_errors=errors,
        )

    def validate_text(self, text: str, request: GenerationRequest) -> list[str]:
        errors = SafetyPolicy(request.safety_mode).validate_response(text)
        audit = self.auditor.audit(text, request.evidence, require_all=False)
        if audit.unknown_ids:
            errors.append(f"Unknown citation ids: {audit.unknown_ids}")
        if not audit.cited_ids and request.evidence.items:
            errors.append("Response must include at least one citation from the evidence packet.")
        if audit.unsupported_sentences:
            errors.append(
                "Long unsupported sentences need citations: "
                + "; ".join(audit.unsupported_sentences[:3])
            )
        return errors

    def _render_prompt(self, request: GenerationRequest) -> str:
        evidence_lines = []
        for item in request.evidence.items:
            locator = item.url or item.path or item.source_id
            evidence_lines.append(
                "\n".join(
                    [
                        f"[{item.citation_id}] {item.title}",
                        f"Source: {locator}",
                        f"Chunk: {item.chunk_id}",
                        f"Support: {item.confidence_note}",
                        f"Snippet: {item.snippet}",
                    ]
                )
            )
        evidence_block = "\n\n".join(evidence_lines) or "No evidence was retrieved."
        return f"""# Codex RAG Task: {request.step}

You are generating academic learning content about fungi. Use only the evidence
packet below. Do not browse, invent sources, or cite anything outside the numbered
evidence items. Cite claims with bracketed source IDs such as [1].

Safety boundary: refuse edibility, dosage, medical decisions, field identification,
and "safe to eat" decisions. Academic discussion of traits, ecology, toxicology,
and risks is allowed when source-backed.

## User Task

{request.task}

## Retrieved Evidence

{evidence_block}

## Output Requirements

- Write Markdown unless the schema requests JSON.
- Include citations from the numbered evidence packet.
- If the evidence is insufficient, say what is missing instead of guessing.
- Keep the answer educational and academic.
"""


class CodexCliGenerator(CodexBridgeGenerator):
    def __init__(self, enabled: bool = False) -> None:
        super().__init__()
        self.enabled = enabled

    def generate(self, request: GenerationRequest) -> GenerationResult:
        result = super().generate(request)
        if not self.enabled or result.status != "pending":
            return result
        codex_path = shutil.which("codex")
        if not codex_path:
            result.validation_errors.append("codex executable was not found on PATH.")
            return result
        try:
            completed = subprocess.run(
                [codex_path, "exec", Path(result.prompt_path).read_text(encoding="utf-8")],
                cwd=str(request.output_dir),
                text=True,
                capture_output=True,
                timeout=180,
                check=False,
            )
            if completed.returncode != 0:
                result.status = "failed"
                result.validation_errors.append(completed.stderr or "codex CLI failed.")
                return result
            Path(result.response_path).write_text(completed.stdout, encoding="utf-8")
            return super().generate(request)
        except Exception as exc:  # noqa: BLE001 - CLI adapter must fail closed.
            result.status = "failed"
            result.validation_errors.append(str(exc))
            return result


def default_response_schema(step: str) -> dict[str, object]:
    return {
        "type": "object",
        "description": f"Optional structured response for {step}. Markdown response is accepted.",
        "properties": {
            "content": {"type": "string"},
            "citations": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["content"],
    }


def build_generator(backend: str = "codex_bridge", enable_codex_cli: bool = False) -> Generator:
    if backend == "codex_cli":
        return CodexCliGenerator(enabled=enable_codex_cli)
    return CodexBridgeGenerator()
