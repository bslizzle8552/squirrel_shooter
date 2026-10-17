"""Synchronous lossless exact-source persistence, independent of AVI admission."""
import hashlib
import json
import os
from time import monotonic
import cv2
import numpy as np
from .recording import RecordingError


def preserve_trip(directory, session_id, packet, observation, *, event_id, reason, index, previous_peak):
    start=monotonic()
    if index>=2048:raise RecordingError('trip_evidence_session_limit')
    image=packet.frame
    if (not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3
        or packet.sequence!=observation['source_sequence'] or packet.generation!=observation['source_generation']
        or packet.received_monotonic!=observation['source_monotonic'] or packet.received_at!=observation['source_wall_time']
        or list(image.shape[:2][::-1])!=observation['image_dimensions'] or not observation['watch_zone_inside']):
        raise RecordingError('trip_evidence_source_mismatch')
    ok,png=cv2.imencode('.png',image,[cv2.IMWRITE_PNG_COMPRESSION,1])
    if not ok:raise RecordingError('trip_evidence_encode_failed')
    dest=directory/session_id
    dest.mkdir(parents=True,exist_ok=True)
    name=f'trip-g{packet.generation}-s{packet.sequence}.png'
    digest=hashlib.sha256(png).hexdigest()
    entry=dict(observation,session_id=session_id,event_id=event_id,reason=reason,filename=name,
        image_sha256=digest,source_pixels_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
        media_role='clean_detector_evidence',encoding='lossless PNG native BGR source',
        evidence_index=index,first=index==0,new_peak=previous_peak is None or observation['confidence']>previous_peak,
        retention_policy='every accepted source; no eviction; 2048 maximum per session; shared storage quota')
    # Exclusive names preserve prior files; rename exposes only fully written bytes.
    for filename,data in [(name,memoryview(png)),(name+'.json',json.dumps(entry,allow_nan=False,indent=2).encode())]:
        path=dest/filename
        if path.exists():raise RecordingError('trip_evidence_identity_exists')
        temp=path.with_name(path.name+'.pending')
        with temp.open('xb') as stream:
            stream.write(data);stream.flush();os.fsync(stream.fileno())
        os.replace(temp,path)
    if os.name!='nt':
        fd=os.open(dest,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    entry['save_latency_seconds']=monotonic()-start
    return entry
