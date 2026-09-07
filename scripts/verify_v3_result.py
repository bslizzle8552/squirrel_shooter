"""Read-only verification of an extracted, user-owned Model A result archive."""
import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path

import torch


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    archive = root.parent.parent / 'runpod_archive' / (root.name + '.tar.gz')
    expected_archive = archive.with_suffix('.gz.sha256').read_text().split()[0]
    assert digest(archive) == expected_archive
    with tarfile.open(archive) as tar:
        archive_files = {}
        for member in tar:
            assert member.isfile() or member.isdir(), member.name
            target = (root.parent / member.name).resolve()
            assert target.is_relative_to(root), member.name
            if member.isfile():
                source_hash = hashlib.file_digest(tar.extractfile(member), 'sha256').hexdigest()
                assert source_hash == digest(target), member.name
                archive_files[member.name] = source_hash
    verified = {}
    for line in (root / 'SHA256SUMS.txt').read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        path = (root / name).resolve()
        assert path.is_relative_to(root), name
        actual = digest(path)
        assert actual == expected, name
        verified[name] = actual
    best = torch.load(root / 'checkpoints/best_ckpt.pth', map_location='cpu', weights_only=False)
    rows = []
    for path in sorted((root / 'checkpoints').glob('*.pth')):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        rows.append(dict(file=path.name, bytes=path.stat().st_size, sha256=digest(path),
                         epoch=checkpoint['start_epoch'], best_ap=float(checkpoint['best_ap']),
                         model_equals_best=all(torch.equal(v, checkpoint['model'][k]) for k, v in best['model'].items())))
    log = (root / 'runtime/model_a_v3_training.log').read_text(encoding='utf-8')
    chronology = []
    for match in re.finditer(r'start train epoch(\d+)(.*?)(?=start train epoch|\Z)', log, re.S):
        metrics = re.findall(r'Average Precision.*?=\s*(-?[\d.]+)\s*$', match[2], re.M)
        if metrics:
            chronology.append(dict(epoch=int(match[1]), ap_metrics_rounded=list(map(float, metrics))))
    epoch = int(best['start_epoch'])
    assert next(x for x in rows if x['file'] == f'epoch_{epoch}_ckpt.pth')['model_equals_best']
    assert not next(x for x in rows if x['file'] == 'epoch_150_ckpt.pth')['model_equals_best']
    report = dict(status='PASS', verified_files=verified, checkpoints=rows,
                  archive_sha256=expected_archive, archive_bytes=archive.stat().st_size,
                  archive_member_hashes=archive_files,
                  best_epoch=epoch, best_ap=float(best['best_ap']),
                  best_sha256=digest(root/'checkpoints/best_ckpt.pth'),
                  best_bytes=(root/'checkpoints/best_ckpt.pth').stat().st_size,
                  checkpoint_keys=list(best), state_keys=list(best['model']),
                  evaluation_chronology=chronology,
                  best_logged_metrics=next(x for x in chronology if x['epoch']==epoch),
                  v2_best_ap=0.6167248023690588,
                  delta=float(best['best_ap'])-0.6167248023690588)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('verified_files','checkpoints','state_keys','evaluation_chronology','archive_member_hashes')}, indent=2))


if __name__ == '__main__':
    main()
