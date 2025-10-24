import unittest
from sigma_lab_v4_2 import SigmaLab, demo_context

class TestSigmaCore(unittest.TestCase):
    def test_diagnose_basic(self):
        cfg, ctx = demo_context("healthcare")
        engine = SigmaLab(cfg)
        result = engine.diagnose(ctx)
        self.assertIn("scores", result)
        self.assertIn("non_harm", result["scores"])
    
    def test_audit_trail(self):
        cfg, ctx = demo_context("healthcare")
        engine = SigmaLab(cfg)
        result = engine.diagnose(ctx)
        audit = engine.export_audit_trail(result)
        self.assertIn("timestamp_utc", audit)

if __name__ == "__main__":
    unittest.main()
