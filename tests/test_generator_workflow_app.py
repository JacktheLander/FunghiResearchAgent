from __future__ import annotations

from pathlib import Path

from fungi_rag.app import build_app
from fungi_rag.evaluate import first_rank_with_terms
from fungi_rag.generator import CodexBridgeGenerator
from fungi_rag.models import EvidenceItem, EvidencePacket, GenerationRequest
from fungi_rag.workflow import APPROVAL_STAGES, FungiWorkflow, workflow_stage_names


def evidence_packet() -> EvidencePacket:
    return EvidencePacket(
        query="hyphae",
        normalized_query="hyphae",
        items=[
            EvidenceItem(
                citation_id=1,
                chunk_id="s1:0",
                source_id="s1",
                title="Fungal morphology",
                snippet="Hyphae increase surface area for absorption.",
                path="fixture.txt",
                fused_score=0.1,
            )
        ],
    )


def test_codex_bridge_writes_packet_and_waits_for_response(tmp_path: Path) -> None:
    request = GenerationRequest(
        run_id="run",
        step="ask",
        task="Answer with evidence.",
        evidence=evidence_packet(),
        output_dir=tmp_path,
    )
    result = CodexBridgeGenerator().generate(request)
    assert result.status == "pending"
    assert Path(result.prompt_path).exists()
    assert Path(result.evidence_path).exists()
    assert Path(result.schema_path).exists()


def test_codex_bridge_validates_citations(tmp_path: Path) -> None:
    request = GenerationRequest(
        run_id="run",
        step="ask",
        task="Answer with evidence.",
        evidence=evidence_packet(),
        output_dir=tmp_path,
    )
    first = CodexBridgeGenerator().generate(request)
    Path(first.response_path).write_text(
        "Hyphae can increase absorptive surface area in fungi [1].",
        encoding="utf-8",
    )
    second = CodexBridgeGenerator().generate(request)
    assert second.status == "accepted"


def test_codex_bridge_rejects_unknown_citation(tmp_path: Path) -> None:
    request = GenerationRequest(
        run_id="run",
        step="ask",
        task="Answer with evidence.",
        evidence=evidence_packet(),
        output_dir=tmp_path,
    )
    first = CodexBridgeGenerator().generate(request)
    Path(first.response_path).write_text("Unsupported citation [9].", encoding="utf-8")
    second = CodexBridgeGenerator().generate(request)
    assert second.status == "invalid"
    assert "Unknown citation ids" in " ".join(second.validation_errors)


def test_workflow_declares_human_approval_stages() -> None:
    stages = workflow_stage_names()
    for stage in APPROVAL_STAGES:
        assert stage in stages


def test_gradio_app_imports_without_api_keys() -> None:
    app = build_app()
    assert hasattr(app, "launch")


def test_first_rank_with_terms_reports_first_relevant_rank() -> None:
    ranked = ["taxonomy overview", "hyphae and mycelium structure", "ecology"]
    assert first_rank_with_terms(ranked, ["mycelium"]) == 2
    assert first_rank_with_terms(ranked, ["not-present"]) == 0


def test_unsafe_ask_refuses_before_retriever_load(monkeypatch, tmp_path: Path) -> None:
    def fail_from_settings(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("retriever should not load for unsafe prompt")

    monkeypatch.setattr("fungi_rag.workflow.HybridRetriever.from_settings", fail_from_settings)
    workflow = FungiWorkflow()
    workflow.settings.output_dir = tmp_path
    result = workflow.ask("Is this mushroom safe to eat?", run_id="unsafe")
    assert result["status"] == "refused"
