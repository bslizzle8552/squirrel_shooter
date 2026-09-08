"""Read-only catalogue of recorder-validated clean files; no video decoding."""
import heapq
import hashlib
import json
from pathlib import Path
import re


class CollectorMedia:
    def __init__(self, directory):
        self.directory = Path(directory)

    def session(self, session_id):
        if not re.fullmatch(r'[0-9a-f]{32}', session_id):
            raise ValueError('invalid session')
        root = self.directory.resolve()
        folder = root / session_id
        path = folder / 'session.json'
        if folder.is_symlink() or path.is_symlink() or path.resolve().parent != folder:
            raise ValueError('unsafe session')
        if path.stat().st_size > 12 * 1024 * 1024:
            raise ValueError('oversized session')
        data = json.loads(path.read_text(encoding='utf-8'))
        if data.get('session_id') != session_id or data.get('file_role') != 'clean_authoritative':
            raise ValueError('unattested session')
        return data

    def file(self, session_id, filename):
        if not re.fullmatch(r'segment-[0-9]{4}\.avi', filename):
            raise ValueError('invalid recording file')
        session = self.session(session_id)
        segment = next((s for s in session.get('segments', []) if s.get('file') == filename), None)
        if (not segment or segment.get('file_role') != 'clean_authoritative'
                or segment.get('status') not in ('complete', 'degraded')
                or not re.fullmatch(r'[0-9a-f]{64}', segment.get('sha256', ''))
                or segment.get('decoded_frames', 0) < 1
                or segment.get('decoded_frames') != segment.get('written_frames')):
            raise ValueError('recording not validated')
        folder = self.directory.resolve() / session_id
        path = folder / filename
        if path.is_symlink() or path.resolve().parent != folder or path.stat().st_size != segment.get('size_bytes'):
            raise ValueError('recording file changed')
        return path, segment

    def trip(self, session_id, filename):
        if not re.fullmatch(r'trip-g[0-9]+-s[0-9]+\.png', filename):
            raise ValueError('invalid trip filename')
        session = self.session(session_id)
        entry = next((v for v in session.get('squirrel_observations', []) if v.get('filename') == filename), None)
        if not entry or entry.get('media_role') != 'clean_detector_evidence' or entry.get('session_id') != session_id:
            raise ValueError('unattested trip')
        folder = self.directory.resolve() / session_id
        path = folder / filename
        if path.is_symlink() or path.resolve().parent != folder or path.stat().st_size > 32*1024*1024:
            raise ValueError('unsafe trip')
        with path.open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest() != entry['image_sha256']:
                raise ValueError('trip image changed')
        return path, entry

    def recent(self, limit=20):
        if not self.directory.exists():
            return []
        candidates = (p for p in self.directory.iterdir()
                      if re.fullmatch(r'[0-9a-f]{32}', p.name) and p.is_dir() and not p.is_symlink())
        folders = heapq.nlargest(limit, candidates, key=lambda p: p.stat().st_mtime)
        rows = []
        for folder in folders:
            try:
                data = self.session(folder.name)
                clips = []
                for segment in data.get('segments', []):
                    try:
                        _, valid = self.file(folder.name, segment.get('file', ''))
                        clips.append({k: valid.get(k) for k in ('file', 'status', 'playback_duration_seconds')})
                    except (OSError, ValueError, TypeError, KeyError):
                        continue
                manual = data.get('manual_requested', False) or any(
                    r.get('action') == 'manual_start_or_extend' for r in data.get('start_reasons', []))
                auto = bool(data.get('qualifying_observation_count')) or any(
                    r.get('reason') == 'squirrel_detection' for r in data.get('source_identities', []))
                trips = [v for v in data.get('squirrel_observations', []) if v.get('media_role') == 'clean_detector_evidence']
                peak = max(trips,key=lambda v:v['confidence']) if trips else None
                rows.append(dict(session_id=folder.name, start_wall=data.get('start_wall'),
                    peak_trip=peak.get('filename') if peak else None,trip_count=len(trips),
                    reason='manual + squirrel_detection' if manual and auto else 'squirrel_detection' if auto else 'manual' if manual else 'unknown',
                    duration_seconds=sum(c.get('playback_duration_seconds') or 0 for c in clips),
                    maximum_squirrel_confidence=data.get('maximum_squirrel_confidence'),
                    status=data.get('status', 'unknown'), error=data.get('error'), clips=clips,
                    file_role='clean_authoritative' if clips else 'unvalidated'))
            except (OSError, ValueError, TypeError, KeyError):
                rows.append(dict(session_id=folder.name, status='unreadable_metadata', clips=[], file_role='unvalidated'))
        return rows
