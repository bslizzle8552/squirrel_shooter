"""Control-incapable composition. Never call app.build_application_runtime/create_app.

Only camera, detector, clean recording and a dedicated diagnostic Flask app are
constructed. No detector-to-recording producer is connected until visit semantics
are defined. Manual recording and the existing in-process extension hook remain.
"""
from __future__ import annotations

import argparse
import secrets
from dataclasses import asdict

from flask import Flask, abort, jsonify, render_template, request

from .camera_service import CameraService
from .config import load_config
from .detector import NcnnSquirrelDetector
from .inference_worker import LatestFrameInferenceWorker
from .recording import RecordingError, RecordingService


class CollectorRuntime:
    """Explicit dependencies; no physical controls or legacy application runtime."""
    def __init__(self, config, *, camera=None, detector=None, recording=None):
        self.config = config
        self.camera = camera if camera is not None else CameraService(
            config.camera, shared_settings=config.shared_camera, encode_jpeg=False,
            frame_buffer_seconds=config.recording.pre_roll_seconds,
            frame_buffer_fps=config.recording.target_fps)
        self.detector = detector if detector is not None else NcnnSquirrelDetector(config.detector)
        self.worker = LatestFrameInferenceWorker(self.camera, self.detector, config.detector)
        self.recording = recording if recording is not None else RecordingService(
            self.camera, config.recording, config.camera.output_directory / 'recordings')
        self._closed = False

    def start(self):
        if self._closed:
            raise RuntimeError('collector already closed')
        try:
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
            self.worker.stop()
        finally:
            try:
                self.recording.stop()
            finally:
                self.camera.stop()

    def status(self):
        return dict(role='control_incapable_collector', physical_control_available=False,
                    automatic_recording_observer_connected=False,
                    detector=self.worker.status(), recording=self.recording.status(),
                    camera=asdict(self.camera.status()))


def create_collector_app(runtime: CollectorRuntime):
    app = Flask(__name__)
    token = secrets.token_urlsafe(32)
    app.config['COLLECTOR_RECORDING_TOKEN'] = token

    @app.get('/')
    def index():
        return render_template('collector.html', recording_token=token)

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
            return jsonify(runtime.recording.record_manual())
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
    try:
        runtime.start()
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    finally:
        runtime.stop()


if __name__ == '__main__':
    main()
