from types import SimpleNamespace
import pytest
from conftest import write_test_config
from squirrel_shooter.config import load_config
from squirrel_shooter.web_dashboard import create_app
from test_recording import rig, captured, finished
from test_web_dashboard import OfflineCameraService, StaticVisionService


def dashboard(tmp_path, rig, monkeypatch):
    import squirrel_shooter.web_dashboard as web
    recorder = rig.build()
    monkeypatch.setattr(web, "build_manual_control_service", lambda *a, **k: pytest.fail("Recording constructed hardware"))
    class NoHardware:
        def status(self):
            return {}
        def __getattr__(self, name):
            pytest.fail("Recording invoked control: " + name)
    app = create_app(app_config=load_config(write_test_config(tmp_path)), camera_service=OfflineCameraService(),
                     vision_service=StaticVisionService(), manual_control_service=NoHardware(), recording_service=recorder,
                     start_camera=False, start_vision=False, temperature_reader=lambda: None)
    app.config["TESTING"] = True
    return app, recorder


def test_manual_api_token_start_stop_and_browser_refresh(tmp_path, rig, monkeypatch):
    app, service = dashboard(tmp_path, rig, monkeypatch)
    client = app.test_client()
    assert client.post("/api/recording/start").status_code == 403
    assert client.post("/api/recording/stop", headers={"X-Control-Token": "wrong"}).status_code == 403
    token = {"X-Control-Token": app.extensions["manual_control_token"]}
    response = client.post("/api/recording/start", headers=token)
    assert response.status_code == 200
    initial = response.json["recording"]
    assert initial["manual_until"] == 130
    captured(rig, service)
    refreshed = app.test_client().get("/api/recording").json["recording"]
    assert refreshed["session_id"] == initial["session_id"] and refreshed["manual_until"] == 130
    page = client.get("/")
    assert b"RECORD 30 SECONDS" in page.data and b"STOP RECORDING" in page.data
    assert b"recording_controls.js" in page.data
    assert client.post("/api/recording/stop", headers=token).status_code == 200
    state = finished(service)
    manifest = client.get("/api/recordings/" + state["session_id"])
    assert manifest.status_code == 200 and manifest.json["file_role"] == "clean_authoritative"
    video = state["segments"][0]["file"]
    assert client.get("/recordings/" + state["session_id"] + "/" + video).status_code == 200
    assert client.get("/recordings/" + state["session_id"] + "/session.json").status_code == 404
    assert client.get("/api/recordings/not-a-session").status_code == 404


def test_stop_api_keeps_automatic_and_polling_does_not_extend(tmp_path, rig, monkeypatch):
    app, service = dashboard(tmp_path, rig, monkeypatch)
    client = app.test_client()
    headers = {"X-Control-Token": app.extensions["manual_control_token"]}
    client.post("/api/recording/start", headers=headers)
    service.extend_automatic_recording(event_id="e", visit_id="v", observed_monotonic=100, reason="future")
    state = client.post("/api/recording/stop", headers=headers).json["recording"]
    assert state["active"] and state["automatic_active"] and state["manual_until"] is None
    rig.clock.now = 102
    for _ in range(3):
        assert client.get("/api/recording").json["recording"]["automatic_reasons"][0]["until"] == 103
    rig.clock.now = 103.1
    finished(service)


def test_unavailable_api_never_starts_anything(tmp_path):
    app = create_app(app_config=load_config(write_test_config(tmp_path)), camera_service=OfflineCameraService(),
                     vision_service=StaticVisionService(), manual_control_service=SimpleNamespace(status=lambda: {}),
                     start_camera=False, start_vision=False)
    client = app.test_client()
    assert client.get("/api/recording").json["recording"]["active"] is False
    for route in ("start", "stop"):
        assert client.post("/api/recording/" + route,
                           headers={"X-Control-Token": app.extensions["manual_control_token"]}).status_code == 503
