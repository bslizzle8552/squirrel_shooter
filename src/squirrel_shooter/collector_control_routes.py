"""Temporary setup endpoints within the existing collector Flask application."""
import secrets

from flask import Response, abort, jsonify, render_template, request

from .collector_calibration import CalibrationError
from .collector_control import ControlError


def register_control_routes(app, control, token, preview_enabled):
    def available():
        if control is None:
            abort(404)
        return control

    @app.get('/api/control')
    def control_status():
        return jsonify(available().status())

    @app.get('/api/control/history')
    def firing_history():
        return jsonify(available().firing_history())

    @app.get('/api/control/history/export')
    def export_firing_history():
        current = available()
        if not current._history_ready:
            abort(503)
        return Response(current.history.export(), mimetype='application/x-ndjson',
                        headers={'Content-Disposition': 'attachment; filename="firing-history.jsonl"'})

    @app.post('/api/control/action')
    def control_action():
        current = available()
        if not secrets.compare_digest(request.headers.get('X-Control-Token', ''), token):
            abort(403)
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or not isinstance(payload.get('action'), str):
            return jsonify(error='JSON action required'), 400
        action = payload.pop('action')
        try:
            result = current.request(action, **payload)
            return jsonify(result), 202 if result['busy'] else 200
        except (ControlError, CalibrationError, TypeError, KeyError, OSError) as exc:
            return jsonify(error=str(exc)), 409

    @app.get('/calibration')
    def calibration_page():
        if not available().status()['calibration_mode']:
            abort(404)
        return render_template('collector_calibration.html', control_token=token,
                               mode='calibration', preview_enabled=preview_enabled)

    @app.get('/aim-test')
    def aim_test_page():
        state = available().status()
        if not state['calibration']['complete'] or state['calibration_mode']:
            abort(404)
        return render_template('collector_calibration.html', control_token=token,
                               mode='test', preview_enabled=preview_enabled)

    @app.get('/control-setup')
    def enter_setup_page():
        available()
        # This deliberate maintenance URL has no preview, polling or automatic action.
        return render_template('collector_control_setup.html', control_token=token)
