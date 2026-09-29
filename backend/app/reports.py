from __future__ import annotations

import re
from datetime import UTC, datetime

from .models import Citation, ConversationDetail


def report_filename(title: str, suffix: str) -> str:
    stem = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "research-report"
    return f"{stem[:80]}.{suffix}"


def citation_location(citation: Citation) -> str:
    parts = [citation.source_type.upper()]
    if citation.page_number:
        parts.append(f"page {citation.page_number}")
    if citation.start_seconds is not None:
        start = format_timestamp(citation.start_seconds)
        end = format_timestamp(citation.end_seconds or citation.start_seconds)
        parts.append(f"{start}-{end}")
    return ", ".join(parts)


def format_timestamp(value: float) -> str:
    seconds = max(0, int(value))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"


def unique_citations(detail: ConversationDetail) -> list[Citation]:
    seen: set[tuple] = set()
    result: list[Citation] = []
    for message in detail.messages:
        for citation in message.citations:
            key = (
                citation.document_id,
                citation.chunk_index,
                citation.page_number,
                citation.start_seconds,
                citation.end_seconds,
            )
            if key not in seen:
                seen.add(key)
                result.append(citation)
    return result


def build_markdown_report(detail: ConversationDetail) -> str:
    lines = [
        f"# {detail.conversation.title}",
        "",
        f"_Research report exported {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}_",
        "",
        "## Findings",
        "",
    ]
    question_number = 0
    for message in detail.messages:
        if message.role == "user":
            question_number += 1
            lines.extend([f"### {question_number}. {message.content}", ""])
            continue
        lines.extend([message.content, ""])
        if message.warning:
            lines.extend([f"> Note: {message.warning}", ""])
        if message.citations:
            lines.extend(["#### Supporting evidence", ""])
            for citation in message.citations:
                passage = citation.passage.replace("\n", " ").strip()
                lines.extend([
                    f"{citation.number}. **{citation.title}** ({citation_location(citation)})",
                    f"   > {passage}",
                    "",
                ])

    evidence = unique_citations(detail)
    lines.extend(["## Evidence register", ""])
    if evidence:
        for index, citation in enumerate(evidence, 1):
            lines.append(
                f"{index}. **{citation.title}** — {citation_location(citation)}; "
                f"document `{citation.document_id}`, passage {citation.chunk_index + 1}"
            )
    else:
        lines.append("No cited evidence was stored with this conversation.")
    lines.extend(["", "---", "", "Generated from a local NexusAI conversation.", ""])
    return "\n".join(lines)


def build_json_report(detail: ConversationDetail) -> dict:
    return {
        "format": "nexusai-research-report-v1",
        "exported_at": datetime.now(UTC).isoformat(),
        "conversation": detail.conversation.model_dump(mode="json"),
        "messages": [message.model_dump(mode="json") for message in detail.messages],
        "evidence_register": [
            citation.model_dump(mode="json") for citation in unique_citations(detail)
        ],
    }
