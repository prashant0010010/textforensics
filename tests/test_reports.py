import json
import re
import unittest

from core.pipeline import run_analysis
from reporting.html_report import build_html
from reporting.json_report import build_json, to_json_string
from reporting.pdf_report import build_pdf
from reporting.text_report import build_text
from tests import samples
from tests.synthetic import make_text, repetitive_text


class TestReports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = repetitive_text(25) + "\n\n" + samples.ACADEMIC + "\n\n" + make_text(1, 800)
        cls.res = run_analysis(cls.text)
        cls.short = run_analysis(samples.SHORT)

    def test_html_sections_and_no_external_resources(self):
        h = build_html(self.res)
        for phrase in ("Executive summary", "Document statistics", "Highlighted text", "Watermark results", "Technical methodology",
                       "Reproducibility", "Human-review recommendations", "Alternative explanations"):
            self.assertIn(phrase, h)
        self.assertIn("not equivalent to", h)
        self.assertIn("Turnitin", h)
        urls = re.findall(r'(?:src|href)=["\'](https?://[^"\']+)', h)
        self.assertEqual(urls, [])
        self.assertNotIn("<link", h)
        self.assertIn("Watermark testing unavailable for this scheme.", h)

    def test_json_valid_strict_and_traceable(self):
        s = to_json_string(self.res)
        data = json.loads(s)                                   # allow_nan=False inside: no NaN/Infinity
        self.assertNotIn("NaN", s)
        self.assertEqual(data["evidence"]["calibrated_probability"], None)
        self.assertEqual(data["document"]["text"], self.res.document.text)
        for sg in data["signals"]:
            self.assertEqual(sg["passage"], self.text[sg["start"]:sg["end"]])
        self.assertTrue(data["reproducibility"]["text_sha256"])
        self.assertFalse(data["reproducibility"]["network_access_used"])

    def test_json_can_omit_text(self):
        d = build_json(self.res, include_text=False)
        self.assertIsNone(d["document"]["text"])
        self.assertTrue(all(sg["passage"] is None for sg in d["signals"]))
        self.assertNotIn(samples.ACADEMIC[:60], json.dumps(d))

    def test_txt_report(self):
        t = build_text(self.res)
        for phrase in ("Executive summary", "Passage-level findings", "Watermark results", "Reproducibility"):
            self.assertIn(phrase, t)
        self.assertIn("not equivalent to", t)

    def test_pdf_report(self):
        pdf = build_pdf(self.res)
        self.assertTrue(pdf.startswith(b"%PDF"))
        self.assertGreater(len(pdf), 5000)

    def test_pdf_handles_unicode_and_highlights_spanning_paragraphs(self):
        res = run_analysis(samples.UNICODE * 30 + "\n\n" + repetitive_text(15))
        self.assertTrue(build_pdf(res).startswith(b"%PDF"))

    def test_reports_for_insufficient_data(self):
        for fn in (build_html, build_text, build_pdf, to_json_string):
            out = fn(self.short)
            self.assertTrue(out)
        self.assertIn("Insufficient data to assess", build_text(self.short))

    def test_reports_never_claim_authorship_or_probability(self):
        for out in (build_html(self.res), build_text(self.res)):
            low = out.lower()
            self.assertNotIn("probability of ai", low.replace("no calibrated probability of ai authorship", ""))
            self.assertNotIn("is ai-generated", low)
            self.assertNotIn("was written by ai", low)

    def test_reports_do_not_write_files(self):
        import os, tempfile
        with tempfile.TemporaryDirectory() as d:
            cwd = os.getcwd()
            os.chdir(d)
            try:
                build_html(self.res), build_text(self.res), to_json_string(self.res), build_pdf(self.res)
                self.assertEqual(os.listdir(d), [])
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
