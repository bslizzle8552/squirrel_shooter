"""Load the sealed real bundle and replay its equivalence inputs through runtime."""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from squirrel_shooter.detector import DetectorConfig,NcnnSquirrelDetector
from squirrel_shooter.model_manifest import ModelManifest,sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--sha256',required=True)
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    manifest=ModelManifest.load(args.manifest,args.sha256)
    detector=NcnnSquirrelDetector(DetectorConfig(enabled=True,manifest_path=str(args.manifest),manifest_sha256=args.sha256))
    assert detector.status()['available'],detector.status()
    evidence=json.loads((args.manifest.parent/'equivalence.json').read_text(encoding='utf-8'))
    rows=[]
    for row in evidence['images']:
        if row['name']=='synthetic_empty_wide': image=np.full((720,1280,3),114,np.uint8)
        elif row['name']=='synthetic_empty_tall': image=np.zeros((640,320,3),np.uint8)
        else:
            provenance=row['provenance']
            assert provenance['final_model_a_role']=='optimizer_train'
            path=args.package/provenance['packaged_filename']
            assert sha256(path)==provenance['packaged_sha256']
            image=cv2.imread(str(path))
        before=image.copy()
        image.setflags(write=False)
        result=detector.infer(image)
        assert result.available and result.identity==manifest.identity
        assert np.array_equal(image,before)
        comparison=row['comparison']
        assert len(result.detections)==comparison['candidate_count']
        for box,pair in zip(result.detections,comparison['pairs']):
            assert abs(box.confidence-pair['candidate']['confidence'])<=.0000005
            assert np.max(np.abs(np.array(box.xyxy)-pair['candidate']['bbox_xyxy']))<=.001
        rows.append(dict(name=row['name'],count=len(result.detections),source_unchanged=True))
    # Fresh reopen checks bytes and readiness again after all real inference.
    assert ModelManifest.load(args.manifest,args.sha256)==manifest
    report=dict(status='PASS',identity=asdict(manifest.identity),images=rows,
                adapter_status=detector.status(),fresh_manifest_reopen=True,
                note='Windows CPU functional validation only; no Pi performance evidence.')
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('PASS: real runtime adapter; sealed manifest; all',len(rows),'equivalence inputs; pristine pixels')


if __name__=='__main__': main()
