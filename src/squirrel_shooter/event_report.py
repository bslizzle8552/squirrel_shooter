"""Build local HTML/Markdown reports and an editable human-review CSV."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import AppConfig, ConfigError, DEFAULT_CONFIG_PATH, load_config


REVIEW_FIELDS = ("event_id", "timestamp", "provisional_category", "human_review_label", "human_review_notes")


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def load_events(root: Path) -> list[dict[str, Any]]:
    events = []
    if root.exists():
        for path in root.rglob("event.json"):
            payload = _load_json(path)
            if payload and payload.get("status") == "complete":
                payload["_event_directory"] = str(path.parent)
                events.append(payload)
    return sorted(events, key=lambda item: str(item.get("start_timestamp", "")))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
            except json.JSONDecodeError:
                continue
    except OSError:
        pass
    return rows


def _existing_reviews(path: Path) -> dict[str, tuple[str, str]]:
    if not path.exists():
        return {}
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return {row["event_id"]: (row.get("human_review_label", ""), row.get("human_review_notes", "")) for row in csv.DictReader(handle) if row.get("event_id")}
    except OSError:
        return {}


def _relative(target: str | Path, report_directory: Path) -> str:
    path = Path(target)
    try:
        return Path(os.path.relpath(path.resolve(), report_directory.resolve())).as_posix()
    except (OSError, ValueError):
        return path.as_posix()


def _storage_used(root: Path) -> int:
    total = 0
    if root.exists():
        for path in root.rglob("*"):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except OSError:
                continue
    return total


def generate_reports(config: AppConfig) -> tuple[Path, Path, Path]:
    """Regenerate all review artifacts from durable event/session files."""

    report_directory = config.reporting.directory
    report_directory.mkdir(parents=True, exist_ok=True)
    review_path = report_directory / "review.csv"
    reviews = _existing_reviews(review_path)
    events = load_events(config.camera.output_directory / "events")
    rejections = _jsonl(config.logging.directory / config.logging.rejection_jsonl)
    session_directory = config.logging.directory / config.logging.sessions_directory
    sessions = [_load_json(path) for path in session_directory.glob("session-*.json")] if session_directory.exists() else []
    sessions = [item for item in sessions if item]

    for event in events:
        label, notes = reviews.get(str(event.get("event_id")), (str(event.get("human_review_label", "")), str(event.get("human_review_notes", ""))))
        event["human_review_label"], event["human_review_notes"] = label, notes
    with review_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for event in events:
            writer.writerow({
                "event_id": event.get("event_id", ""),
                "timestamp": event.get("start_timestamp", ""),
                "provisional_category": event.get("provisional_category", ""),
                "human_review_label": event.get("human_review_label", ""),
                "human_review_notes": event.get("human_review_notes", ""),
            })

    categories = Counter(str(event.get("provisional_category", "unclassified_motion")) for event in events)
    rejection_counts = Counter(str(item.get("reason", "global_motion_unclassified")) for item in rejections)
    by_hour = Counter(str(event.get("start_timestamp", ""))[11:13] or "unknown" for event in events)
    event_fps = [float(event.get("measured_camera_fps", 0) or 0) for event in events if float(event.get("measured_camera_fps", 0) or 0) > 0]
    runtime = 0.0
    dropped = 0
    camera_errors = 0
    for session in sessions:
        dropped += int(session.get("dropped_or_failed_frame_reads", 0) or 0)
        camera_errors += int(session.get("camera_read_errors", 0) or 0)
        try:
            start, end = datetime.fromisoformat(session["startup_time"]), datetime.fromisoformat(session["shutdown_time"])
            runtime += max(0.0, (end - start).total_seconds())
        except (KeyError, TypeError, ValueError):
            continue
    session_average_fps = [float(item.get("average_measured_fps", 0) or 0) for item in sessions if float(item.get("average_measured_fps", 0) or 0) > 0]
    session_minimum_fps = [float(item.get("minimum_measured_fps", 0) or 0) for item in sessions if float(item.get("minimum_measured_fps", 0) or 0) > 0]
    session_maximum_fps = [float(item.get("maximum_measured_fps", 0) or 0) for item in sessions if float(item.get("maximum_measured_fps", 0) or 0) > 0]
    average_fps = session_average_fps or event_fps
    minimum_fps = session_minimum_fps or event_fps
    maximum_fps = session_maximum_fps or event_fps
    period = "No completed events"
    if events:
        period = f"{events[0].get('start_timestamp')} through {events[-1].get('end_timestamp')}"
    summary = {
        "Reporting period": period,
        "Total recorded runtime": f"{runtime / 3600:.2f} hours",
        "Confirmed events": len(events),
        "Rejected global-motion events": len(rejections),
        "Average measured FPS": f"{sum(average_fps) / len(average_fps):.2f}" if average_fps else "unknown",
        "Minimum measured FPS": f"{min(minimum_fps):.2f}" if minimum_fps else "unknown",
        "Maximum measured FPS": f"{max(maximum_fps):.2f}" if maximum_fps else "unknown",
        "Dropped/failed-frame warnings": dropped,
        "Camera-read errors": camera_errors,
        "Capture storage used": f"{_storage_used(config.camera.output_directory) / (1024 * 1024):.2f} MB",
    }

    html_path = report_directory / "latest-report.html"
    css = """
body{font-family:system-ui,sans-serif;max-width:1200px;margin:auto;padding:1.5rem;background:#f6f7f4;color:#1b261c}
table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #ccd3ca;padding:.45rem;text-align:left}
.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(__THUMBNAIL__px,1fr));gap:1rem}.card{background:white;border:1px solid #ccd3ca;border-radius:9px;padding:1rem}
.card img{width:100%;height:220px;object-fit:contain;background:#111}.muted{color:#566158}.warning{background:#fff4d6;padding:.8rem;border-radius:6px}
""".replace("__THUMBNAIL__", str(config.reporting.thumbnail_width))
    cards: list[str] = []
    for event in events:
        snapshot = _relative(str(event.get("snapshot_path", "")), report_directory)
        clip = _relative(str(event.get("clip_path", "")), report_directory)
        attributes = ", ".join(event.get("movement_attributes", [])) or "none"
        cards.append(f"""<article class="card">
<a href="{html.escape(snapshot)}"><img src="{html.escape(snapshot)}" alt="Event {html.escape(str(event.get('event_id')))}"></a>
<h3>{html.escape(str(event.get('start_timestamp', '')))}</h3>
<p><strong>{html.escape(str(event.get('provisional_category', '')))}</strong> <span class="muted">(heuristic only)</span></p>
<p>Movement: {html.escape(attributes)}<br>Duration: {float(event.get('duration', 0)):.2f}s; travel: {float(event.get('total_centroid_travel', 0)):.1f}px; average speed: {float(event.get('average_pixel_speed', 0)):.1f}px/s<br>Components: {int(event.get('component_count', 0))}; grouping confidence: {float(event.get('grouping_confidence', 0)):.2f}</p>
<p><a href="{html.escape(snapshot)}">Full snapshot</a> | <a href="{html.escape(clip)}">Video clip</a></p>
<p>Human label: <strong>{html.escape(str(event.get('human_review_label') or '(blank)'))}</strong><br>Review notes: {html.escape(str(event.get('human_review_notes') or '(blank)'))}</p>
</article>""")
    def count_table(title: str, counts: Counter[str]) -> str:
        rows = "".join(f"<tr><td>{html.escape(key)}</td><td>{value}</td></tr>" for key, value in sorted(counts.items())) or "<tr><td>None</td><td>0</td></tr>"
        return f"<h2>{html.escape(title)}</h2><table><tr><th>Type</th><th>Count</th></tr>{rows}</table>"
    summary_rows = "".join(f"<tr><th>{html.escape(key)}</th><td>{html.escape(str(value))}</td></tr>" for key, value in summary.items())
    html_path.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>Squirrel Squirter Event Review</title><style>{css}</style></head><body>
<h1>Garden Motion Review</h1><p class="warning">All categories are provisional motion/size heuristics, not species recognition. Use <a href="review.csv">review.csv</a> for human labels.</p>
<table>{summary_rows}</table>{count_table('Events by provisional category', categories)}{count_table('Global rejections by reason', rejection_counts)}{count_table('Events by hour', by_hour)}
<h2>Chronological event gallery</h2><section class="gallery">{''.join(cards) or '<p>No completed events yet.</p>'}</section></body></html>""", encoding="utf-8")

    markdown_path = report_directory / "latest-report.md"
    lines = ["# Garden Motion Review", "", "> Categories are provisional heuristics, not species recognition.", "", "## Summary", ""]
    lines.extend(f"- **{key}:** {value}" for key, value in summary.items())
    for title, counts in (("Events by provisional category", categories), ("Global rejections by reason", rejection_counts), ("Events by hour", by_hour)):
        lines.extend(["", f"## {title}", ""])
        lines.extend(f"- {key}: {value}" for key, value in sorted(counts.items()))
        if not counts:
            lines.append("- None")
    lines.extend(["", "## Events", ""])
    for event in events:
        snapshot = _relative(str(event.get("snapshot_path", "")), report_directory)
        clip = _relative(str(event.get("clip_path", "")), report_directory)
        lines.extend([
            f"### {event.get('start_timestamp', '')} - {event.get('provisional_category', '')}", "",
            f"- Movement: {', '.join(event.get('movement_attributes', [])) or 'none'}",
            f"- Duration: {float(event.get('duration', 0)):.2f}s; travel: {float(event.get('total_centroid_travel', 0)):.1f}px; average speed: {float(event.get('average_pixel_speed', 0)):.1f}px/s",
            f"- Components: {event.get('component_count', 0)}; grouping confidence: {float(event.get('grouping_confidence', 0)):.2f}",
            f"- [Snapshot]({snapshot}) | [Video clip]({clip})",
            f"- Human review: {event.get('human_review_label') or '(blank)'} - {event.get('human_review_notes') or '(blank)'}", "",
        ])
    if not events:
        lines.append("No completed events yet.")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return html_path, markdown_path, review_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rebuild local motion-event review reports")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser


def main() -> int:
    try:
        config = load_config(build_parser().parse_args().config)
        paths = generate_reports(config)
    except (ConfigError, OSError) as exc:
        print(f"Report error: {exc}", file=sys.stderr)
        return 1
    for path in paths:
        print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
