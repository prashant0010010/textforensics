import ast
import logging
import os
import socket
import tempfile
import unittest
from unittest import mock

from config.settings import AnalysisSettings
from core.ingestion import from_docx_bytes, from_pdf_bytes
from core.pipeline import run_analysis
from reporting.html_report import build_html
from reporting.json_report import to_json_string
from reporting.pdf_report import build_pdf
from reporting.text_report import build_text
from tests.synthetic import make_text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NETWORK_MODULES = {"socket", "requests", "urllib.request", "urllib3", "http.client", "httpx", "aiohttp", "ftplib", "smtplib",
                   "telnetlib", "websocket", "websockets", "xmlrpc.client", "poplib", "imaplib", "paramiko", "grpc"}
SKIP_DIRS = {"tests", ".git", "__pycache__", ".venv", "venv"}


def project_sources():
    for dp, dn, fn in os.walk(ROOT):
        dn[:] = [d for d in dn if d not in SKIP_DIRS]
        for f in fn:
            if f.endswith(".py"):
                yield os.path.join(dp, f)


class TestPrivacyAndOffline(unittest.TestCase):
    def test_no_network_library_is_imported_anywhere_in_the_application(self):
        offenders = []
        for path in project_sources():
            with open(path, encoding="utf-8") as fh:
                tree = ast.parse(fh.read())
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
                for n in names:
                    if n in NETWORK_MODULES or n.split(".")[0] in {"requests", "urllib3", "httpx", "aiohttp"}:
                        offenders.append((os.path.relpath(path, ROOT), n))
        self.assertEqual(offenders, [])

    def test_full_analysis_works_with_network_disabled(self):
        def boom(*a, **k):
            raise AssertionError("network access attempted")
        with mock.patch.object(socket, "socket", boom), mock.patch.object(socket, "create_connection", boom), \
                mock.patch.object(socket, "getaddrinfo", boom):
            res = run_analysis(make_text(9, 2000))
            build_html(res), build_text(res), to_json_string(res), build_pdf(res)
        self.assertFalse(res.reproducibility["network_access_used"])
        self.assertFalse([s for s in res.statuses if s.status == "failed"])

    def test_logs_never_contain_submitted_text(self):
        marker = "Zyxwvutsr quixotic marmalade lighthouse"
        text = marker + ". " + make_text(4, 1500)
        records = []

        class H(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        root = logging.getLogger()
        h = H()
        old = root.level
        root.addHandler(h)
        root.setLevel(logging.DEBUG)
        try:
            import core.pipeline as p

            def failing(doc, cfg, ctx):
                raise KeyError(marker)          # message deliberately contains document text
            patched = [(n, sw, (failing if n == "entropy" else f)) for n, sw, f in p.PIPELINE]
            with mock.patch.object(p, "PIPELINE", patched):
                res = p.run_analysis(text, AnalysisSettings())
        finally:
            root.removeHandler(h)
            root.setLevel(old)
        self.assertTrue(any(s.status == "failed" for s in res.statuses))      # failure isolated, analysis continued
        self.assertFalse([r for r in records if "Zyxwvutsr" in r])
        self.assertTrue(any("module entropy failed" in r for r in records))

    def test_analysis_writes_no_files_and_leaves_no_temp_files(self):
        with tempfile.TemporaryDirectory() as work, tempfile.TemporaryDirectory() as tmp:
            cwd, old_tmp = os.getcwd(), tempfile.tempdir
            os.chdir(work)
            tempfile.tempdir = tmp
            try:
                res = run_analysis(make_text(3, 1500))
                to_json_string(res), build_html(res), build_text(res), build_pdf(res)
                self.assertEqual(os.listdir(work), [])
                self.assertEqual(os.listdir(tmp), [])
            finally:
                os.chdir(cwd)
                tempfile.tempdir = old_tmp

    def test_streamlit_telemetry_disabled_in_config(self):
        with open(os.path.join(ROOT, ".streamlit", "config.toml"), encoding="utf-8") as fh:
            self.assertIn("gatherUsageStats = false", fh.read())

    def test_launcher_disables_telemetry(self):
        with open(os.path.join(ROOT, "run_windows.bat"), encoding="utf-8") as fh:
            self.assertIn("STREAMLIT_BROWSER_GATHER_USAGE_STATS=false", fh.read())

    def test_malformed_uploads_fail_safely(self):
        from core.ingestion import IngestionError
        for fn, data in ((from_docx_bytes, b"PK\x03\x04junk"), (from_pdf_bytes, b"\x00\x01garbage")):
            with self.assertRaises(IngestionError):
                fn(data)


if __name__ == "__main__":
    unittest.main()
