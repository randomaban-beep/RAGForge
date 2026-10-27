from streamlit.testing.v1 import AppTest
from pathlib import Path


def test_sample_demo_and_clear():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app/streamlit_app.py").run(
        timeout=30
    )
    assert not app.exception
    app.sidebar.button[0].click().run(timeout=30)
    assert not app.exception
    app.chat_input[0].set_value("How often are API keys rotated?").run(timeout=30)
    assert not app.exception
    messages = app.session_state["messages"]
    assert "90 days" in messages[-1]["content"]
    assert messages[-1]["result"]["citations"] == ["keys"]
    app.sidebar.button[2].click().run(timeout=30)
    assert app.session_state["pipeline"].index is None
    assert app.session_state["messages"] == []
