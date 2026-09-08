"""Control-incapable composition. Never call app.build_application_runtime/create_app.

Only camera, detector, clean recording and a dedicated collector Flask app are
constructed. Observations record evidence without asserting biological visits.
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


class CollectorRuntime:
    """Explicit dependencies; no physical controls or legacy application runtime."""
    def __init__(self, config, *, camera=None, detector=None, recording=None):
        self.config = config
        cv2.setNumThreads(config.runtime.opencv_threads)
        self.camera = camera if camera is not None else CameraService(
            config.camera, shared_settings=config.shared_camera, encode_jpeg=config.collector_preview.enabled,
            jpeg_quality=config.collector_preview.jpeg_quality,
            preview_maximum_width=config.collector_preview.maximum_width,
            frame_buffer_seconds=config.recording.pre_roll_seconds,
            frame_buffer_fps=config.recording.target_fps)
        self.detector = detector if detector is not None else NcnnSquirrelDetector(config.detector)
        self.recording = recording if recording is not None else RecordingService(
            self.camera, replace(config.recording, automatic_tail_seconds=config.automatic_recording.tail_seconds),
            config.camera.output_directory / 'recordings')
        self.observer = SquirrelRecordingObserver(self.recording, config.automatic_recording,
                                                  config.detector.maximum_result_age_seconds)
        self.worker = LatestFrameInferenceWorker(self.camera, self.detector, config.detector,
                                                 observer=self.observer.observe)
        self.media = CollectorMedia(config.camera.output_directory / 'recordings')
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
                    automatic_recording_observer_connected=self.config.automatic_recording.enabled,
                    automatic_recording=self.observer.status(),
                    preview=asdict(self.config.collector_preview),
                    detector=self.worker.status(), recording=self.recording.status(compact=True),
                    camera=asdict(self.camera.status()))


def create_collector_app(runtime: CollectorRuntime):
    app = Flask(__name__)
    token = secrets.token_urlsafe(32)
    app.config['COLLECTOR_RECORDING_TOKEN'] = token

    @app.get('/')
    def index():
        return render_template('collector.html', recording_token=token,
                               preview_enabled=runtime.config.collector_preview.enabled)

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
