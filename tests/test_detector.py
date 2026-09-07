import copy
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from squirrel_shooter.detector import DetectorConfig, NcnnSquirrelDetector, decode, preprocess
from squirrel_shooter.model_manifest import CONTRACT, ModelManifest, sha256


@pytest.fixture
def bundle(tmp_path):
    for name in ('model.param', 'model.bin'):
        (tmp_path/name).write_bytes(name.encode())
    report = dict(status='PASS', checkpoint_sha256='a'*64,
                  artifact_hashes={key:sha256(tmp_path/name) for key,name in [('param','model.param'),('bin','model.bin')]})
    (tmp_path/'equivalence.json').write_text(json.dumps(report))
    data = copy.deepcopy(CONTRACT)
    data.update(model_id='test', model_version='v1', source_training_run='run', source_checkpoint_sha256='a'*64,
                exported_at_utc='2026-09-07T00:00:00Z', export_tools={'ncnn':'1.0.20260526'},
                equivalence_report_sha256=sha256(tmp_path/'equivalence.json'),
                artifacts={key:dict(file=name, bytes=(tmp_path/name).stat().st_size, sha256=sha256(tmp_path/name))
                           for key,name in [('param','model.param'),('bin','model.bin'),('equivalence','equivalence.json')]})
    path = tmp_path/'manifest.json'
    def seal():
        path.write_text(json.dumps(data))
        return sha256(path)
    seal()
    return path, data, seal


@pytest.mark.parametrize('key,value', [
    ('classes',['chipmunk']), ('classes',['squirrel','person']), ('class_count',True),
    ('ready',False), ('schema_version',2), ('intended_role','auto_fire'),
    ('input',{}), ('output',{}), ('runtime',{}), ('export_status','FAIL'), ('equivalence_status','FAIL'),
])
def test_manifest_rejects_wrong_contract(bundle, key, value):
    path,data,seal = bundle
    data[key]=value
    with pytest.raises(ValueError):
        ModelManifest.load(path,seal())


def test_manifest_pin_and_artifact_hashes(bundle):
    path,data,seal=bundle
    pin=seal()
    assert ModelManifest.load(path,pin).identity.model_id=='test'
    with pytest.raises(ValueError,match='manifest hash'):
        ModelManifest.load(path,'b'*64)
    (path.parent/'model.bin').write_bytes(b'wrong!!!!')
    with pytest.raises(ValueError,match='artifact hash'):
        ModelManifest.load(path,pin)


def test_manifest_rejects_evidence_for_other_model(bundle):
    path,data,seal=bundle
    data['source_checkpoint_sha256']='c'*64
    with pytest.raises(ValueError,match='checkpoint'):
        ModelManifest.load(path,seal())


def test_manifest_rejects_duplicate_unknown_and_nonfinite_keys(bundle):
    path,data,seal=bundle
    data['unknown']=True
    with pytest.raises(ValueError,match='schema'):
        ModelManifest.load(path,seal())
    del data['unknown']
    data['export_tools']['bad']=float('nan')
    with pytest.raises(ValueError,match='nonfinite'):
        ModelManifest.load(path,seal())
    del data['export_tools']['bad']
    seal()
    path.write_text(path.read_text().replace('"schema_version": 1','"schema_version": 1, "schema_version": 1'))
    with pytest.raises(ValueError,match='duplicate'):
        ModelManifest.load(path,sha256(path))


def test_published_schema_constants_match_runtime():
    schema=json.loads((Path(__file__).resolve().parents[1]/'docs/model-manifest-v1.schema.json').read_text())
    assert schema['additionalProperties'] is False
    for key,value in CONTRACT.items():
        assert schema['properties'][key]['const']==value


@pytest.mark.parametrize('name',['../model.bin','C:\\model.bin','/model.bin'])
def test_manifest_path_rejection(bundle,name):
    path,data,seal=bundle
    data['artifacts']['bin']['file']=name
    with pytest.raises(ValueError):
        ModelManifest.load(path,seal())


def test_preprocessing_exact_bgr_top_left_floor_resize_and_immutable():
    frame=np.arange(37*71*3,dtype=np.uint8).reshape(37,71,3)
    before=frame.copy()
    frame.setflags(write=False)
    tensor,ratio=preprocess(frame)
    resized=cv2.resize(frame,(416,int(37*ratio)),interpolation=cv2.INTER_LINEAR)
    assert tensor.dtype==np.float32 and tensor.flags.c_contiguous
    assert np.array_equal(tensor[:,:resized.shape[0]].transpose(1,2,0),resized)
    assert np.all(tensor[:,resized.shape[0]:]==114)
    assert np.array_equal(frame,before)
    tensor[:]=0
    assert np.array_equal(frame,before)


def test_native_inverse_and_low_confidence_nms():
    raw=np.zeros((3549,6),np.float32)
    raw[0]=[10,10,np.log(4),np.log(2),.5,.02]
    boxes=decode(raw,.5,832,416,.01)
    assert len(boxes)==1
    assert boxes[0].xyxy==pytest.approx((128,144,192,176))
    assert boxes[0].confidence==pytest.approx(.01)
    raw[1]=[9,10,np.log(4),np.log(2),.5,.02]
    assert len(decode(raw,.5,832,416,.01))==1
    raw[:,5]=.01999
    assert decode(raw,.5,832,416,.01)==()


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf')])
def test_nonfinite_rejected(value):
    raw=np.zeros((3549,6),np.float32)
    raw[0,0]=value
    with pytest.raises(ValueError):
        decode(raw,1,416,416,.01)


def test_empty_invalid_probabilities_and_exponential_overflow():
    raw=np.zeros((3549,6),np.float32)
    assert decode(raw,1,416,416,.01)==()
    raw[0,4]=1.01
    with pytest.raises(ValueError): decode(raw,1,416,416,.01)
    raw[0,4]=0
    raw[0,2]=1000
    with pytest.raises(FloatingPointError): decode(raw,1,416,416,.01)


def fake_backend(raw=None,code=0,load=0):
    class Mat:
        def __init__(self,array): self.array=array
        def clone(self): return self
    class Net:
        def __init__(self): self.opt=SimpleNamespace()
        def load_param(self,p):
            assert self.opt.use_vulkan_compute is False
            assert self.opt.use_fp16_packed is False
            assert self.opt.use_fp16_storage is False
            assert self.opt.use_fp16_arithmetic is False
            assert self.opt.use_bf16_storage is False
            assert self.opt.use_int8_inference is False
            return load
        def load_model(self,p): return load
        def create_extractor(self): return self
        def input(self,name,data):
            assert name=='in0' and data.array.shape==(3,416,416)
            return 0
        def extract(self,name):
            assert name=='out0'
            return code, raw if raw is not None else np.zeros((3549,6),np.float32)
    return SimpleNamespace(__version__='1.0.20260526',Net=Net,Mat=Mat)


@pytest.mark.parametrize('failure',['none','load','extract','nonfinite','shape','version','manifest'])
def test_adapter_returns_explicit_state_and_identity(bundle,failure):
    path,data,seal=bundle
    raw=np.zeros((3549,6),np.float32)
    if failure=='nonfinite': raw[0,0]=np.nan
    if failure=='shape': raw=raw[:1]
    backend=fake_backend(raw,code=-1 if failure=='extract' else 0,load=-1 if failure=='load' else 0)
    if failure=='version': backend.__version__='other'
    pin=seal() if failure!='manifest' else 'f'*64
    adapter=NcnnSquirrelDetector(DetectorConfig(enabled=True,manifest_path=str(path),manifest_sha256=pin),backend_module=backend)
    frame=np.zeros((720,1280,3),np.uint8)
    frame.setflags(write=False)
    output=adapter.infer(frame)
    assert output.available == (failure=='none')
    assert output.identity is not None if failure!='manifest' else output.identity is None
    assert np.count_nonzero(frame)==0
    if failure!='none': assert output.error


@pytest.mark.parametrize('values',[{'target_hz':float('nan')},{'num_threads':True},{'diagnostic_confidence_floor':0},
                                   {'enabled':1},{'backend':'caffe'},{'maximum_source_age_seconds':-1}, {'dog':True}])
def test_detector_config_rejects_invalid_values(values):
    with pytest.raises((ValueError,TypeError)): DetectorConfig(**values)
