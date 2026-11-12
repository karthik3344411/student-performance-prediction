"""Smoke test for the Streamlit app: it loads, the form submits and a prediction is shown."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.predict import MODEL_PATH

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="model not trained")
def test_app_predicts():
    at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value.endswith("Student Performance Prediction System")

    at.button[0].click().run()  # the form's "Predict final grade" button
    assert not at.exception
    predicted = [m.value for m in at.metric if m.label == "Predicted final grade (G3)"]
    assert predicted, "prediction metric not shown"
    grade = float(predicted[0].split("/")[0])
    assert 0 <= grade <= 20
