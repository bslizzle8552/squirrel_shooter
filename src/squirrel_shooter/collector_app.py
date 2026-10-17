"""Collector composition. Never call app.build_application_runtime/create_app.

Camera, detector and recording remain collector-owned. Optional physical control
is separately enabled and consumes the existing detector observations.
"""
from __future__ import annotations

import argparse
import secrets
from dataclasses import asdict, replace
import signal

import cv2
from flask import Flask, Response, abort, jsonify, render_template, request, send_file

from .camera_service import CameraService
from .config import load_config
from .detector import NcnnSquirrelDetector
from .inference_worker import LatestFrameInferenceWorker
from .recording import RecordingError, RecordingService
from .collector_policy import SquirrelRecordingObserver
from .collector_media import CollectorMedia
from .collector_night import NightVisionMonitor


class CollectorRuntime:
    """Explicit dependencies; legacy application services are never constructed."""
    def __init__(self, config, *, camera=None, detector=None, recording=None):
        self.config = config
        cv2.setNumThreads(config.runtime.opencv_threads)
        self.camera = camera if camera is not None else CameraService(
            config.camera, shared_settings=config.shared_camera, encode_jpeg=config.collector_preview.enabled,
            jpeg_quality=config.collector_preview.jpeg_quality,
            preview_maximum_width=config.collector_preview.maximum_width,
            preview_overlay=config.watch_zone.overlay,
            frame_buffer_seconds=config.recording.pre_roll_seconds,
            frame_buffer_fps=config.recording.target_fps)
        self.detector = detector if detector is not None else NcnnSquirrelDetector(config.detector)
        self.recording = recording if recording is not None else RecordingService(
            self.camera, replace(config.recording, automatic_tail_seconds=config.automatic_recording.tail_seconds,
                                 observation_max_age_seconds=config.detector.maximum_result_age_seconds),
            config.camera.output_directory / 'recordings')
        self.observer = SquirrelRecordingObserver(self.recording, config.automatic_recording,
                                                  config.detector.maximum_result_age_seconds, watch_zone=config.watch_zone,
                                                  require_evidence=True)
        self.control = None
        self.worker = LatestFrameInferenceWorker(self.camera, self.detector, config.detector,
                                                 frame_observer=self._observe if config.physical_control.enabled else self.observer.observe,
                                                 frame_monitor=self._monitor_frame if config.collector_night.enabled else None)
        self.night = NightVisionMonitor(config.collector_night)
        self.worker.set_paused(self.night.status()['paused'], self._night_transition)
        if config.physical_control.enabled:
            from .collector_control import CollectorControl
            self.control = CollectorControl(config.physical_control, config.automatic_fire,
                camera_status=self.camera.status, detector_status=self.worker.status,
                watch_zone=config.watch_zone, maximum_age=config.detector.maximum_result_age_seconds,
                recording=self.recording)
        self.media = CollectorMedia(config.camera.output_directory / 'recordings')
        self._closed = False

    def _observe(self, observation, packet):
        # Preserve exact-frame recording behavior; actuators only receive a small result copy.
        try:
            self.observer.observe(observation, packet)
        finally:
            if self.control is not None:
                self.control.observe(observation)

    def _night_transition(self, paused):
        self.observer.set_paused(paused)
        self.recording.set_idle_requested(paused)

    def _monitor_frame(self, packet):
        if not self.night.sample(packet):
            return
        paused = self.night.status()['paused']
        self.worker.set_paused(paused, self._night_transition)
        # Retain old pre-roll until any already-admitted session has drained.
        recording = self.recording.status(compact=True)
        self.camera.set_buffering_enabled(not paused or recording['active'] or recording['pending_sessions'] > 0)

    def start(self):
        if self._closed:
            raise RuntimeError('collector already closed')
        try:
            if self.control is not None:
                self.control.start()
            self.camera.start()
            self.recording.start()
            self.worker.start()
        except Exception:
            self.stop()
            raise

    def stop(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self.control is not None:
                self.control.stop()
        finally:
            try:
                self.worker.stop()
            finally:
                try:
                    self.recording.stop()
                finally:
                    self.camera.stop()

    def status(self):
        control = self.control.status(compact=True) if self.control is not None else None
        return dict(role='collector_with_optional_control' if control else 'control_incapable_collector',
                    physical_control_available=bool(control and control['worker_alive'] and not control['error']),
                    control=control,
                    automatic_recording_observer_connected=self.config.automatic_recording.enabled,
                    automatic_recording=self.observer.status(),
                    night_mode=self.night.status(),
                    watch_zone=dict(**asdict(self.config.watch_zone),sha256=self.config.watch_zone.sha256),
                    preview=asdict(self.config.collector_preview),
                    detector=self.worker.status(), recording=self.recording.status(compact=True),
                    camera=asdict(self.camera.status()))


def create_collector_app(runtime: CollectorRuntime):
    app = Flask(__name__)
    token = secrets.token_urlsafe(32)
    app.config['COLLECTOR_RECORDING_TOKEN'] = token
    app.config['COLLECTOR_CONTROL_TOKEN'] = token
    if runtime.control is not None:
        from .collector_control_routes import register_control_routes
        register_control_routes(app, runtime.control, token, runtime.config.collector_preview.enabled)

    @app.get('/')
    def index():
        return render_template('collector.html', recording_token=token,
                               preview_enabled=runtime.config.collector_preview.enabled,
                               control_enabled=runtime.control is not None)

    @app.get('/preview.mjpg')
    def preview():
        if not runtime.config.collector_preview.enabled:
            abort(404)
        return Response(runtime.camera.mjpeg_frames(maximum_fps=runtime.config.collector_preview.maximum_fps,
                                                    annotated_only=False),
                        mimetype='multipart/x-mixed-replace; boundary=frame',
                        headers={'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'})

    @app.get('/api/recordings')
    def recordings():
        return jsonify(recordings=runtime.media.recent())

    def clean_file(session_id, filename):
        try:
            return runtime.media.file(session_id, filename)
        except (OSError, ValueError, TypeError, KeyError):
            abort(404)

    @app.get('/recordings/<session_id>/<filename>')
    def recording_file(session_id, filename):
        path, segment = clean_file(session_id, filename)
        return send_file(path, conditional=True, etag=segment['sha256'],
                         as_attachment=request.args.get('download') == '1',
                         download_name=f'{session_id}-{filename}', mimetype='video/x-msvideo')

    @app.get('/trip-evidence/<session_id>/<filename>')
    def trip_file(session_id, filename):
        try:
            path, entry = runtime.media.trip(session_id,filename)
        except (OSError, ValueError, TypeError, KeyError):
            abort(404)
        return send_file(path, mimetype='image/png', conditional=True, etag=entry['image_sha256'])

    @app.get('/clips/<session_id>/<filename>')
    def clip(session_id, filename):
        _, segment = clean_file(session_id, filename)
        return render_template('collector_clip.html', session_id=session_id, filename=filename,
                               segment=segment, session_status=runtime.media.session(session_id).get('status', 'unknown'))

    @app.get('/api/status')
    def status():
        return jsonify(runtime.status())

    @app.get('/api/detector')
    def detector_status():
        return jsonify(runtime.worker.status())

    @app.get('/api/recording')
    def recording_status():
        return jsonify(runtime.recording.status())

    def authorize():
        if not secrets.compare_digest(request.headers.get('X-Recording-Token', ''), token):
            abort(403)

    @app.post('/api/recording/start')
    def record_start():
        authorize()
        try:
            return jsonify(runtime.worker.while_unpaused(runtime.recording.record_manual))
        except RecordingError as exc:
            return jsonify(error=str(exc)), 409

    @app.post('/api/recording/stop')
    def record_stop():
        authorize()
        return jsonify(runtime.recording.stop_manual())

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/default.yaml')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=5094, type=int)
    args = parser.parse_args()
    runtime = CollectorRuntime(load_config(args.config))
    app = create_collector_app(runtime)
    def terminate(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    try:
        runtime.start()
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    finally:
        runtime.stop()


if __name__ == '__main__':
    main()
