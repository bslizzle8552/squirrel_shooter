"""Offline v3 export gate. Uses v2 tolerances and non-protected training pixels."""
import argparse
import csv
import importlib.util
import json
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import ncnn
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from squirrel_shooter.detector import decode, preprocess
from squirrel_shooter.model_manifest import CONTRACT, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--ml-root', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    args = parser.parse_args()
    ml, work = args.ml_root, args.work
    result = ml/'07_models/model_a_v3_yolox_nano/extracted_results/model_a_v3_yolox_nano_20260907T210507Z'
    checkpoint = result/'checkpoints/best_ckpt.pth'
    assert sha256(checkpoint) == '5d8b78244107302881477bda95353236f29356c23a2a10f61936225c4dfc4831'
    source = ml/'07_models/model_a_yolox_nano_v1/evaluation/local_environment/yolox-src'
    sys.path.insert(0, str(source))
    from yolox.exp import get_exp
    from yolox.utils import postprocess
    exp = get_exp(str(result/'configs/model_a_v3_yolox_nano.py'), None)
    assert exp.ema and exp.num_classes == 1 and exp.test_size == (416, 416)
    model = exp.get_model()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'], strict=True)
    model.eval()
    model.head.decode_in_inference = False
    torch.set_num_threads(2)
    spec = importlib.util.spec_from_file_location('v2_equivalence', ml/'08_evaluation/pi4_benchmark/_tooling/scripts/compare_export_equivalence.py')
    v2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v2)
    package = ml/'06_training/packages/model_a_v3_yolox_nano'
    rows = list(csv.DictReader((package/'manifests/package_manifest.csv').open(encoding='utf-8-sig')))
    groups = defaultdict(list)
    for row in rows:
        if row['final_model_a_role'] == 'optimizer_train' and row['ready_to_train'] == 'true':
            if row['contribution_class'] == 'external_train_real':
                groups[(row['source_dataset'], row['squirrel_positive_vs_negative'])].append(row)
            elif row['source_dataset'] == 'Approved banked native local squirrel evidence since Model A v2':
                groups[('native', 'positive')].append(row)
    selected = [r for key in sorted(groups) for r in sorted(groups[key], key=lambda x:x['packaged_filename'])[:4]]
    (work/'equivalence_images').mkdir(exist_ok=True)
    images = []
    for row in selected:
        path = package / row['packaged_filename']
        assert '/train/images/' in row['packaged_filename']
        assert sha256(path) == row['packaged_sha256']
        images.append((path.name, cv2.imread(str(path)), row))
    images += [('synthetic_empty_wide', np.full((720, 1280, 3), 114, np.uint8), {'role':'synthetic'}),
               ('synthetic_empty_tall', np.zeros((640, 320, 3), np.uint8), {'role':'synthetic'})]
    net = ncnn.Net()
    net.opt.num_threads = 2
    net.opt.use_vulkan_compute = False
    assert net.load_param(str(work/'export/v3.ncnn.param')) == 0
    assert net.load_model(str(work/'export/v3.ncnn.bin')) == 0
    reports = []
    for name, image, provenance in images:
        assert image is not None
        before = image.copy()
        image.setflags(write=False)
        tensor, ratio = preprocess(image)
        assert np.array_equal(tensor, v2.val_transform(image))
        with torch.inference_mode():
            raw = model(torch.from_numpy(tensor[None])).numpy()
        ex = net.create_extractor()
        assert ex.input('in0', ncnn.Mat(tensor).clone()) == 0
        code, output = ex.extract('out0')
        assert code == 0
        converted = np.array(output, dtype=np.float32)
        assert raw.shape == (1, 3549, 6) and converted.shape == (3549, 6)
        reference = v2.detections_from_decoded(v2.decode_yolox(raw), ratio, image.shape[1], image.shape[0], postprocess)
        candidate = [{'bbox_xyxy': list(b.xyxy), 'confidence': b.confidence} for b in decode(converted, ratio, image.shape[1], image.shape[0], .01)]
        comparison = v2.compare_detections(reference, candidate)
        assert np.array_equal(image, before)
        reports.append(dict(name=name, provenance=provenance, comparison=comparison,
                            raw_max_abs_delta=float(np.max(np.abs(raw[0]-converted))),
                            source_unchanged=True, preprocessing_exact=True))
    # Edge logic is a postprocessing contract, not model/threshold selection.
    edge = np.zeros((3549, 6), np.float32)
    edge[0] = [10, 10, 1, 1, .5, .02]
    assert len(decode(edge, 1, 416, 416, .01)) == 1
    edge[0, 5] = .01999
    assert not decode(edge, 1, 416, 416, .01)
    pairs = [p for r in reports for p in r['comparison']['pairs']]
    summary = dict(images=len(reports), all_counts_match=all(r['comparison']['count_match'] for r in reports),
                   maximum_confidence_delta=max(p['confidence_absolute_delta'] for p in pairs),
                   maximum_box_delta=max(p['maximum_box_coordinate_absolute_delta'] for p in pairs),
                   minimum_iou=min(p['box_iou'] for p in pairs),
                   no_detection_cases=sum(r['comparison']['reference_count']==0 for r in reports),
                   low_confidence_pairs=sum(.01 <= p['reference']['confidence'] < .1 for p in pairs))
    passed = all(r['comparison']['pass'] for r in reports) and summary['no_detection_cases'] > 0 and summary['low_confidence_pairs'] > 0
    report = dict(status='PASS' if passed else 'FAIL', checkpoint_sha256=sha256(checkpoint),
                  artifact_hashes={k:sha256(work/f'export/v3.ncnn.{ext}') for k,ext in [('param','param'),('bin','bin')]},
                  tolerance=v2.TOLERANCE, summary=summary, images=reports,
                  evaluated_at_utc=datetime.now(timezone.utc).isoformat(),
                  environment=dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
                                   ncnn=ncnn.__version__, numpy=np.__version__, opencv=cv2.__version__),
                  boundaries='Optimizer training samples only; no locked validation or Golden images; no model selection or threshold tuning.',
                  low_confidence_edge_contract='0.010000 included; 0.009995 excluded',
                  source_config_sha256=sha256(result/'configs/model_a_v3_yolox_nano.py'))
    (work/'export_equivalence.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit('MATERIAL EQUIVALENCE FAILURE: integration prohibited')
    # Seal a new versioned directory only after the gate passed.
    import shutil
    args.bundle.mkdir(parents=True, exist_ok=False)
    for ext in ('param', 'bin'):
        shutil.copy2(work/f'export/v3.ncnn.{ext}', args.bundle/f'model.ncnn.{ext}')
    shutil.copy2(work/'export_equivalence.json', args.bundle/'equivalence.json')
    manifest = dict(CONTRACT, model_id='model_a_yolox_nano', model_version='v3-epoch149-fp32-416-r1',
                    source_training_run=result.name, source_checkpoint_sha256=sha256(checkpoint),
                    exported_at_utc=report['evaluated_at_utc'], export_tools=report['environment'] | {'pnnx':'20260526'},
                    equivalence_report_sha256=sha256(args.bundle/'equivalence.json'),
                    artifacts={key:dict(file=name, bytes=(args.bundle/name).stat().st_size, sha256=sha256(args.bundle/name))
                               for key,name in [('param','model.ncnn.param'),('bin','model.ncnn.bin'),('equivalence','equivalence.json')]})
    path = args.bundle/'manifest.json'
    path.write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    (args.bundle/'manifest.sha256').write_text(sha256(path)+'  manifest.json\n')
    print('SEALED', str(args.bundle), sha256(path))


if __name__ == '__main__':
    main()
