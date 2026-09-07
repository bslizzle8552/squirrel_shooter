"""Strict, versioned FP32 raw YOLOX deployment contract; no filename readiness."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path


CONTRACT = {
    'schema_version': 1,
    'classes': ['squirrel'], 'class_count': 1,
    'input': {'width': 416, 'height': 416, 'color_order': 'BGR',
              'resize': 'aspect_ratio_min_floor_linear', 'letterbox': 'top_left',
              'fill': 114, 'layout': 'CHW', 'dtype': 'float32',
              'scale': 1.0, 'mean': [0, 0, 0], 'std': [1, 1, 1], 'name': 'in0'},
    'output': {'name': 'out0', 'shape': [3549, 6], 'decoded': False,
               'fields': ['grid_x', 'grid_y', 'log_w', 'log_h', 'objectness', 'class_score'],
               'strides': [8, 16, 32], 'confidence': 'objectness_times_class',
               'nms': 'class_agnostic_greedy', 'nms_iou': 0.65,
               'box_coordinates': 'native_xyxy_clipped'},
    'runtime': {'backend': 'ncnn', 'version': '1.0.20260526', 'precision': 'fp32', 'device': 'cpu'},
    'export_status': 'PASS', 'equivalence_status': 'PASS',
    'intended_role': 'control_incapable_collector', 'ready': True,
}


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _hash(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('invalid SHA-256')


@dataclass(frozen=True)
class ModelIdentity:
    model_id: str
    model_version: str
    checkpoint_sha256: str
    manifest_sha256: str
    param_sha256: str
    bin_sha256: str


@dataclass(frozen=True)
class ModelManifest:
    identity: ModelIdentity
    param_path: Path
    bin_path: Path

    @classmethod
    def load(cls, path: Path, expected_sha256: str) -> ModelManifest:
        """A separately pinned manifest digest binds readiness and all artifact bytes."""
        _hash(expected_sha256)
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError('manifest hash mismatch')
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError('duplicate manifest key')
                result[key] = value
            return result
        def invalid_constant(value):
            raise ValueError(f'nonfinite JSON value: {value}')
        data = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid_constant)
        extra = {'model_id', 'model_version', 'source_training_run', 'source_checkpoint_sha256',
                 'artifacts', 'exported_at_utc', 'export_tools', 'equivalence_report_sha256'}
        if set(data) != set(CONTRACT) | extra:
            raise ValueError('manifest schema keys mismatch')
        for key, value in CONTRACT.items():
            # JSON typed equality avoids accepting true for 1, or false for 0.
            if json.dumps(data[key], sort_keys=True) != json.dumps(value, sort_keys=True):
                raise ValueError(f'unsupported manifest contract: {key}')
        for key in ('model_id', 'model_version', 'source_training_run', 'exported_at_utc'):
            if not isinstance(data[key], str) or not data[key].strip():
                raise ValueError(f'invalid {key}')
        _hash(data['source_checkpoint_sha256'])
        _hash(data['equivalence_report_sha256'])
        if not isinstance(data['export_tools'], dict) or not data['export_tools']:
            raise ValueError('missing export identity')
        artifacts = data['artifacts']
        if set(artifacts) != {'param', 'bin', 'equivalence'}:
            raise ValueError('artifact schema mismatch')
        paths = {}
        for key, artifact in artifacts.items():
            if set(artifact) != {'file', 'sha256', 'bytes'}:
                raise ValueError('artifact schema mismatch')
            _hash(artifact['sha256'])
            name = artifact['file']
            if not isinstance(name, str) or '/' in name or '\\' in name or ':' in name or name in ('', '.', '..'):
                raise ValueError('artifact must be a bundle basename')
            target = (path.parent / name).resolve()
            if target.parent != path.parent.resolve():
                raise ValueError('artifact escapes bundle')
            if type(artifact['bytes']) is not int or artifact['bytes'] <= 0:
                raise ValueError('invalid artifact size')
            if target.stat().st_size != artifact['bytes'] or sha256(target) != artifact['sha256']:
                raise ValueError(f'artifact hash/size mismatch: {key}')
            paths[key] = target
        if artifacts['equivalence']['sha256'] != data['equivalence_report_sha256']:
            raise ValueError('equivalence identity mismatch')
        report = json.loads(paths['equivalence'].read_text(encoding='utf-8'))
        if report.get('status') != 'PASS' or report.get('checkpoint_sha256') != data['source_checkpoint_sha256']:
            raise ValueError('equivalence not passed for checkpoint')
        for key in ('param', 'bin'):
            if report.get('artifact_hashes', {}).get(key) != artifacts[key]['sha256']:
                raise ValueError('equivalence not bound to model artifacts')
        return cls(ModelIdentity(data['model_id'], data['model_version'], data['source_checkpoint_sha256'],
                                 expected_sha256, artifacts['param']['sha256'], artifacts['bin']['sha256']),
                   paths['param'], paths['bin'])
