from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

import httpx
import yaml

from fungi_rag.config import Settings, get_settings
from fungi_rag.models import SourceManifest, SourceManifestEntry
from fungi_rag.utils import append_jsonl, ensure_dir, sha256_bytes, slugify, utc_now_iso


DEFAULT_MANIFEST = Path("examples/source_manifest.yaml")


def load_manifest(path: Path | str = DEFAULT_MANIFEST) -> SourceManifest:
    return SourceManifest.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


class SourceDownloader:
    def __init__(self, settings: Settings | None = None, rate_limit_seconds: float = 0.75) -> None:
        self.settings = settings or get_settings()
        self.rate_limit_seconds = rate_limit_seconds
        ensure_dir(self.settings.source_raw_dir)

    def download_manifest(self, manifest: SourceManifest, refresh: bool = False) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        with httpx.Client(follow_redirects=True, timeout=30.0) as client:
            for entry in manifest.sources:
                rows.append(self.download_entry(client, entry, refresh=refresh))
                time.sleep(self.rate_limit_seconds)
        append_jsonl(self.settings.source_state_path, rows)
        return rows

    def download_entry(
        self,
        client: httpx.Client,
        entry: SourceManifestEntry,
        *,
        refresh: bool = False,
    ) -> dict[str, object]:
        extension = extension_for(entry)
        output_path = self.settings.source_raw_dir / f"{slugify(entry.id)}{extension}"
        sidecar_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
        if output_path.exists() and sidecar_path.exists() and not refresh:
            return {
                "id": entry.id,
                "title": entry.title,
                "url": entry.url,
                "local_path": str(output_path),
                "status": "skipped_existing",
                "retrieved_at": utc_now_iso(),
            }
        try:
            response = client.get(entry.url, headers={"User-Agent": "fungi-rag/0.1 academic downloader"})
            response.raise_for_status()
            content = response.content
            output_path.write_bytes(content)
            checksum = sha256_bytes(content)
            metadata = {
                "id": entry.id,
                "title": entry.title,
                "url": str(response.url),
                "source_type": entry.source_type,
                "license_note": entry.license_note,
                "topics": entry.topics,
                "checksum": checksum,
                "retrieved_at": utc_now_iso(),
                "http_status": response.status_code,
            }
            sidecar_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            return {
                **metadata,
                "local_path": str(output_path),
                "status": "downloaded",
            }
        except Exception as exc:  # noqa: BLE001 - preserve downloader progress across source failures.
            return {
                "id": entry.id,
                "title": entry.title,
                "url": entry.url,
                "status": "failed",
                "error": str(exc),
                "retrieved_at": utc_now_iso(),
            }


def extension_for(entry: SourceManifestEntry) -> str:
    if entry.source_type == "pdf":
        return ".pdf"
    if entry.source_type == "markdown":
        return ".md"
    if entry.source_type == "text":
        return ".txt"
    return ".html"


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download open fungi academic source corpus.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    download = subparsers.add_parser("download")
    download.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    download.add_argument("--refresh", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "download":
        manifest = load_manifest(Path(args.manifest))
        rows = SourceDownloader().download_manifest(manifest, refresh=args.refresh)
        downloaded = sum(1 for row in rows if row.get("status") == "downloaded")
        failed = sum(1 for row in rows if row.get("status") == "failed")
        skipped = sum(1 for row in rows if row.get("status") == "skipped_existing")
        print(f"Downloaded {downloaded}, skipped {skipped}, failed {failed}")


if __name__ == "__main__":
    main()
