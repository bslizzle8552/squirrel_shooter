"""Small device-local nine-point map; no camera, hardware, or background work.

Four convex cells use the inverse-bilinear method from the earlier manual
calibration implementation. Completed maps are validated and cached once.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile
import threading

from .pan_tilt import PanTiltConfig, PanTiltPosition, clamp_angle


class CalibrationError(ValueError):
    """Calibration is invalid, incomplete, or unsuitable for this target."""


@dataclass(frozen=True)
class _Point:
    point: int
    pixel_x: float
    pixel_y: float
    pan: float
    tilt: float


_CELLS = ((1, 2, 5, 4), (2, 3, 6, 5), (4, 5, 8, 7), (5, 6, 9, 8))
_BOUNDARY = (1, 2, 3, 6, 9, 8, 7, 4)
_EPS = 1e-9


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationError(f'{name} must be finite')
    try:
        number = float(value)
    except OverflowError as exc:
        raise CalibrationError(f'{name} must be finite') from exc
    if not math.isfinite(number):
        raise CalibrationError(f'{name} must be finite')
    return number


def _dimensions(width, height):
    if any(isinstance(v, bool) or not isinstance(v, int) or not 0 < v <= 2**31 - 1
           for v in (width, height)):
        raise CalibrationError('Camera dimensions must be positive 32-bit integers')
    return width, height


def _point_number(point):
    if isinstance(point, bool) or not isinstance(point, int) or not 1 <= point <= 9:
        raise CalibrationError('Point must be an integer from 1 to 9')
    return point


def _xy(point):
    return point.pixel_x, point.pixel_y


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a, b, p):
    return (abs(_cross(a, b, p)) <= _EPS
            and min(a[0], b[0]) - _EPS <= p[0] <= max(a[0], b[0]) + _EPS
            and min(a[1], b[1]) - _EPS <= p[1] <= max(a[1], b[1]) + _EPS)


def _intersects(a, b, c, d):
    x, y, z, w = _cross(a, b, c), _cross(a, b, d), _cross(c, d, a), _cross(c, d, b)
    return ((x * y < 0 and z * w < 0) or _on_segment(a, b, c) or _on_segment(a, b, d)
            or _on_segment(c, d, a) or _on_segment(c, d, b))


def _area(polygon):
    return sum(a[0] * b[1] - b[0] * a[1]
               for a, b in zip(polygon, polygon[1:] + polygon[:1])) / 2


def _inside_polygon(target, polygon):
    inside = False
    x, y = target
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        if _on_segment(a, b, target):
            return True
        if (a[1] > y) != (b[1] > y):
            crossing = (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]
            if x < crossing:
                inside = not inside
    return inside


def _inside_convex(target, polygon, *, strict=False):
    sign = 1 if _area(polygon) > 0 else -1
    values = [sign * _cross(a, b, target) for a, b in zip(polygon, polygon[1:] + polygon[:1])]
    return all(value > _EPS for value in values) if strict else all(value >= -_EPS for value in values)


def _validated_cells(points):
    indexed = {point.point: point for point in points}
    if len(points) != 9 or set(indexed) != set(range(1, 10)):
        raise CalibrationError('Complete calibration requires all nine points')
    if len({_xy(point) for point in points}) != 9:
        raise CalibrationError('Calibration contains duplicate pixels')
    boundary = tuple(_xy(indexed[number]) for number in _BOUNDARY)
    winding = _area(boundary)
    if abs(winding) <= _EPS:
        raise CalibrationError('Calibration boundary is degenerate')
    # Include internal edges, so folds/intersections cannot be hidden by a
    # simple outer boundary. Shared endpoints are allowed; shared pixels are not.
    edges = sorted({tuple(sorted((cell[i], cell[(i + 1) % 4]))) for cell in _CELLS for i in range(4)})
    for i, (a, b) in enumerate(edges):
        for c, d in edges[i + 1:]:
            if len({a, b, c, d}) == 4 and _intersects(_xy(indexed[a]), _xy(indexed[b]),
                                                    _xy(indexed[c]), _xy(indexed[d])):
                raise CalibrationError('Calibration grid is folded or has intersecting edges')
    if not _inside_polygon(_xy(indexed[5]), boundary):
        raise CalibrationError('Calibration center is outside its boundary')
    cells = tuple(tuple(indexed[number] for number in cell) for cell in _CELLS)
    polygons = [tuple(map(_xy, cell)) for cell in cells]
    sign = 1 if winding > 0 else -1
    for polygon in polygons:
        turns = [sign * _cross(polygon[i], polygon[(i + 1) % 4], polygon[(i + 2) % 4]) for i in range(4)]
        if not all(turn > _EPS for turn in turns):
            raise CalibrationError('Calibration cells must be convex, nondegenerate, and consistently oriented')
    for i, polygon in enumerate(polygons):
        for other in polygons[i + 1:]:
            if any(_inside_convex(vertex, other, strict=True) for vertex in polygon):
                raise CalibrationError('Calibration cells overlap')
            if any(_inside_convex(vertex, polygon, strict=True) for vertex in other):
                raise CalibrationError('Calibration cells overlap')
    return cells


def _bilinear(values, u, v):
    a, b, c, d = values
    return (1 - u) * (1 - v) * a + u * (1 - v) * b + u * v * c + (1 - u) * v * d


def _inverse(x, y, cell):
    """Bounded Newton solve from the proven four-corner calibration mapping."""
    xs = tuple(point.pixel_x for point in cell)
    ys = tuple(point.pixel_y for point in cell)
    u = v = 0.5
    for _ in range(20):
        ex, ey = x - _bilinear(xs, u, v), y - _bilinear(ys, u, v)
        if max(abs(ex), abs(ey)) < 1e-7:
            break
        dxdu = (1 - v) * (xs[1] - xs[0]) + v * (xs[2] - xs[3])
        dxdv = (1 - u) * (xs[3] - xs[0]) + u * (xs[2] - xs[1])
        dydu = (1 - v) * (ys[1] - ys[0]) + v * (ys[2] - ys[3])
        dydv = (1 - u) * (ys[3] - ys[0]) + u * (ys[2] - ys[1])
        determinant = dxdu * dydv - dxdv * dydu
        if abs(determinant) < 1e-12:
            raise CalibrationError('Calibration cell cannot be inverted')
        u += (ex * dydv - ey * dxdv) / determinant
        v += (dxdu * ey - dydu * ex) / determinant
        if not math.isfinite(u) or not math.isfinite(v):
            raise CalibrationError('Calibration interpolation is not finite')
    if (max(abs(x - _bilinear(xs, u, v)), abs(y - _bilinear(ys, u, v))) > 1e-5
            or not (-_EPS <= u <= 1 + _EPS and -_EPS <= v <= 1 + _EPS)):
        raise CalibrationError('Target cannot be mapped inside its calibrated cell')
    return min(max(u, 0.0), 1.0), min(max(v, 0.0), 1.0)


class CalibrationStore:
    """Atomic explicit-lock/complete workflow for one device calibration file."""

    def __init__(self, path: Path, limits: PanTiltConfig):
        self.path = Path(path)
        self.limits = limits
        self._lock = threading.Lock()
        self._size = None
        self._points = ()
        self._complete = False
        self._cells = ()
        self._error = None
        try:
            if self.path.exists():
                raw = json.loads(self.path.read_text(encoding='utf-8'))
                if (not isinstance(raw, dict) or isinstance(raw.get('schema_version'), bool)
                        or raw.get('schema_version') != 1):
                    raise CalibrationError('Unsupported calibration file schema')
                size = _dimensions(raw.get('camera_width'), raw.get('camera_height'))
                if not isinstance(raw.get('complete'), bool) or not isinstance(raw.get('points'), list):
                    raise CalibrationError('Calibration needs a complete flag and points list')
                points = tuple(self._validated_point(**record, width=size[0], height=size[1]) for record in raw['points'])
                if len({p.point for p in points}) != len(points):
                    raise CalibrationError('Calibration contains duplicate point numbers')
                cells = _validated_cells(points) if raw['complete'] else ()
                self._size = size
                self._points = tuple(sorted(points, key=lambda point: point.point))
                self._complete, self._cells = raw['complete'], cells
        except (OSError, ValueError, TypeError, OverflowError) as exc:
            self._error = f'{type(exc).__name__}: {exc}; explicit reset is required'

    def _validated_point(self, point, pixel_x, pixel_y, pan, tilt, width, height):
        point = _point_number(point)
        x, y, pan, tilt = (_number(value, name) for value, name in
                           ((pixel_x, 'pixel_x'), (pixel_y, 'pixel_y'), (pan, 'pan'), (tilt, 'tilt')))
        if not 0 <= x < width or not 0 <= y < height:
            raise CalibrationError('Calibration pixel must be inside the native camera dimensions')
        if not self.limits.pan_min <= pan <= self.limits.pan_max or not self.limits.tilt_min <= tilt <= self.limits.tilt_max:
            raise CalibrationError('Calibration angles must be inside configured servo limits')
        return _Point(point, x, y, pan, tilt)

    def _require_usable(self, width, height):
        size = _dimensions(width, height)
        if self._error:
            raise CalibrationError(self._error)
        if self._size is not None and size != self._size:
            raise CalibrationError('Camera dimensions mismatch; explicitly reset for a new calibration')
        return size

    def _save(self, size, points, complete):
        # Validation, candidate state and disk replacement precede cache commit.
        cells = _validated_cells(points) if complete else ()
        ordered = tuple(sorted(points, key=lambda point: point.point))
        payload = dict(schema_version=1, camera_width=size[0], camera_height=size[1],
                       complete=complete, points=[asdict(point) for point in ordered])
        encoded = (json.dumps(payload, allow_nan=False, indent=2) + '\n').encode('utf-8')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f'.{self.path.name}.', suffix='.tmp', dir=self.path.parent)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
        self._size, self._points, self._complete, self._cells = size, ordered, complete, cells
        self._error = None

    def status(self):
        with self._lock:
            return dict(camera_width=self._size[0] if self._size else None,
                        camera_height=self._size[1] if self._size else None,
                        complete=self._complete, points=[asdict(point) for point in self._points],
                        error=self._error)

    def lock_point(self, point, pixel_x, pixel_y, pan, tilt, width, height, *, relock=False):
        with self._lock:
            size = self._require_usable(width, height)
            if not isinstance(relock, bool):
                raise CalibrationError('Relock must be explicitly true or false')
            record = self._validated_point(point, pixel_x, pixel_y, pan, tilt, *size)
            existing = {item.point: item for item in self._points}
            if record.point in existing and not relock:
                raise CalibrationError('Point is already locked; explicitly select RELOCK to replace it')
            if relock and record.point not in existing:
                raise CalibrationError('Cannot relock a point that is not locked')
            existing[record.point] = record
            self._save(size, tuple(existing.values()), False)
        return self.status()

    def complete(self, width, height):
        with self._lock:
            size = self._require_usable(width, height)
            self._save(size, self._points, True)
        return self.status()

    def reset(self, width, height):
        """Explicitly discard prior points; also recovers an invalid stored file."""
        size = _dimensions(width, height)
        with self._lock:
            self._save(size, (), False)
        return self.status()

    def aim(self, pixel_x, pixel_y, width, height):
        with self._lock:
            self._require_usable(width, height)
            x, y = _number(pixel_x, 'pixel_x'), _number(pixel_y, 'pixel_y')
            if not self._complete:
                raise CalibrationError('Calibration is incomplete')
            if not 0 <= x < width or not 0 <= y < height:
                raise CalibrationError('Target is outside the native camera dimensions')
            for point in self._points:
                if (x, y) == _xy(point):
                    return PanTiltPosition(point.pan, point.tilt)
            for cell in self._cells:
                if _inside_convex((x, y), tuple(map(_xy, cell))):
                    u, v = _inverse(x, y, cell)
                    pan = _bilinear(tuple(point.pan for point in cell), u, v)
                    tilt = _bilinear(tuple(point.tilt for point in cell), u, v)
                    return PanTiltPosition(clamp_angle(pan, self.limits.pan_min, self.limits.pan_max),
                                           clamp_angle(tilt, self.limits.tilt_min, self.limits.tilt_max))
            raise CalibrationError('Target is outside the calibrated region')
