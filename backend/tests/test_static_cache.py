from fastapi.testclient import TestClient

from app.main import app


def test_ui_assets_are_revalidated_so_edits_show_up_without_a_hard_refresh():
    client = TestClient(app)
    for path in ("/app/css/app.css", "/app/js/runs.js", "/app/"):
        response = client.get(path)
        assert response.status_code == 200, path
        assert response.headers["cache-control"] == "no-cache", path
        assert "etag" in response.headers or "last-modified" in response.headers, path
