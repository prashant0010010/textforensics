"""Executes app.py against a minimal FAKE Streamlit (the real package may not be installed where tests run).

This validates the application's own logic and wiring, not Streamlit's rendering."""
import os
import runpy
import sys
import types
import unittest
from unittest import mock

from tests.synthetic import make_text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _Any:
    def __getattr__(self, name):
        return lambda *a, **k: _Any()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Proxy(_Any):
    """Column/tab/expander stand-in: widget calls are routed to the fake Streamlit (so they return real values)."""

    def __init__(self, fake):
        self._fake = fake

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        if name in FakeStreamlit.__dict__:
            return getattr(self._fake, name)
        return lambda *a, **k: _Any()


class _State(dict):
    __getattr__ = dict.get

    def __setattr__(self, k, v):
        self[k] = v


class FakeStreamlit(types.ModuleType):
    def __init__(self, text="", press=("Start analysis", "Prepare downloads")):
        super().__init__("streamlit")
        self.text, self.press = text, set(press)
        self.session_state = _State()
        self.sidebar = self
        self.calls = []
        self.html_calls = []
        comp = types.ModuleType("streamlit.components")
        v1 = types.ModuleType("streamlit.components.v1")
        v1.html = lambda h, height=None, scrolling=False: self.html_calls.append(h)
        comp.v1 = v1
        self.components = comp
        self._mods = {"streamlit": self, "streamlit.components": comp, "streamlit.components.v1": v1}

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *a, **k: (self.calls.append((name, a)), _Any())[1]

    def cache_data(self, *a, **k):
        return lambda f: f

    def checkbox(self, label, value=False, **k):
        return value

    def selectbox(self, label, options, index=0, **k):
        return options[index]

    def number_input(self, label, *a, **k):
        return a[2] if len(a) >= 3 else a[0]

    def text_input(self, label, value="", **k):
        return value

    def text_area(self, label, **k):
        return self.text

    def file_uploader(self, *a, **k):
        return None

    def button(self, label, **k):
        return label in self.press and not k.get("disabled", False)

    def columns(self, n, **k):
        return [_Proxy(self) for _ in range(n if isinstance(n, int) else len(n))]

    def tabs(self, names):
        return [_Proxy(self) for _ in names]

    def expander(self, *a, **k):
        return _Proxy(self)

    def progress(self, *a, **k):
        return _Any()

    def empty(self):
        return _Any()


def run_app(fake):
    with mock.patch.dict(sys.modules, fake._mods):
        runpy.run_path(os.path.join(ROOT, "app.py"), run_name="__main__")


class TestAppSmoke(unittest.TestCase):
    def test_no_input_shows_no_results(self):
        fake = FakeStreamlit("")
        run_app(fake)
        self.assertIsNone(fake.session_state.get("result"))

    def test_analysis_flow_and_downloads(self):
        fake = FakeStreamlit(make_text(21, 2500))
        run_app(fake)
        res = fake.session_state["result"]
        self.assertEqual(res.doc_stats["words_total"] > 2000, True)
        self.assertTrue(any("sig-" in h or "textpane" in h for h in fake.html_calls))     # highlighted panel rendered
        d = fake.session_state["downloads"]
        self.assertTrue(d["html"].startswith(b"<!doctype html>"))
        self.assertTrue(d["json"].startswith(b"{"))
        self.assertIn("Executive summary", d["txt"].decode())
        self.assertTrue(d["pdf"].startswith(b"%PDF"))

    def test_short_text_gives_insufficient_label(self):
        fake = FakeStreamlit("Just a few words here, nothing more to see.", press=("Start analysis",))
        run_app(fake)
        self.assertEqual(fake.session_state["result"].evidence["category"], 5)


if __name__ == "__main__":
    unittest.main()
