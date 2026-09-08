"""Native normalized site geometry; presentation overlays never modify sources."""
from dataclasses import dataclass
import hashlib
import json
import math
import cv2
import numpy as np


@dataclass(frozen=True)
class WatchZoneConfig:
    version: str = 'full-frame-default'
    polygon: tuple = ((0.,0.),(1.,0.),(1.,1.),(0.,1.))

    def __post_init__(self):
        if not isinstance(self.version,str) or not 1 <= len(self.version) <= 128:
            raise ValueError('invalid watch zone version')
        if not isinstance(self.polygon,(list,tuple)) or not 3 <= len(self.polygon) <= 128:
            raise ValueError('watch zone needs 3..128 vertices')
        pts=[]
        for p in self.polygon:
            if not isinstance(p,(list,tuple)) or len(p)!=2 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1 for v in p):
                raise ValueError('watch zone coordinates must be finite normalized pairs')
            pts.append(tuple(float(v) for v in p))
        if len(set(pts))!=len(pts):raise ValueError('duplicate watch vertices')
        if abs(sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(pts,pts[1:]+pts[:1])))<1e-8:
            raise ValueError('degenerate watch zone')
        # Reject crossings and nonadjacent edge touches, including collinear overlaps.
        def orient(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
        def on(a,b,c):return abs(orient(a,b,c))<1e-12 and min(a[0],b[0])<=c[0]<=max(a[0],b[0]) and min(a[1],b[1])<=c[1]<=max(a[1],b[1])
        edges=list(zip(pts,pts[1:]+pts[:1]))
        for i,(a,b) in enumerate(edges):
            for j,(c,d) in enumerate(edges):
                if j<=i+1 or (i==0 and j==len(edges)-1):continue
                if (orient(a,b,c)*orient(a,b,d)<0 and orient(c,d,a)*orient(c,d,b)<0) or on(a,b,c) or on(a,b,d) or on(c,d,a) or on(c,d,b):
                    raise ValueError('self intersecting watch zone')
        object.__setattr__(self,'polygon',tuple(pts))

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(dict(version=self.version,polygon=self.polygon),sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def contains(self,x,y):
        if not math.isfinite(x) or not math.isfinite(y) or not 0<=x<=1 or not 0<=y<=1:return False
        inside=False
        for a,b in zip(self.polygon,self.polygon[1:]+self.polygon[:1]):
            cross=(x-a[0])*(b[1]-a[1])-(y-a[1])*(b[0]-a[0])
            if abs(cross)<=1e-10 and min(a[0],b[0])-1e-10<=x<=max(a[0],b[0])+1e-10 and min(a[1],b[1])-1e-10<=y<=max(a[1],b[1])+1e-10:return True
            if (a[1]>y)!=(b[1]>y) and x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0]:inside=not inside
        return inside

    def test_box(self,box,width,height):
        if len(box)!=4 or width<=0 or height<=0 or any(not math.isfinite(v) for v in box) or not 0<=box[0]<box[2]<=width or not 0<=box[1]<box[3]<=height:
            raise ValueError('invalid native watch box')
        point=((box[0]+box[2])/2,box[3])
        return dict(watch_zone_test_point=list(point),watch_zone_inside=self.contains(point[0]/width,point[1]/height),watch_zone_version=self.version,watch_zone_sha256=self.sha256)

    def pixels(self,width,height):
        return np.array([(min(width-1,round(x*width)),min(height-1,round(y*height))) for x,y in self.polygon],dtype=np.int32)

    def overlay(self,frame):
        view=frame.copy()
        cv2.polylines(view,[self.pixels(frame.shape[1],frame.shape[0])],True,(0,0,255),1,cv2.LINE_AA)
        return view
