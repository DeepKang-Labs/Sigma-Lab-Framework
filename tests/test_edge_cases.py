import unittest
from sigma_lab_v4_2 import SigmaLab, demo_context, OptionContext

class TestEdgeCases(unittest.TestCase):
    def test_invalid_context(self):
        cfg, _ = demo_context("public")
        ctx = OptionContext(name="bad", short_term_risk=1.5, long_term_risk=-0.3,
                            irreversibility_risk=0.5, stakeholders=[])
        engine = SigmaLab(cfg)
        result = engine.diagnose(ctx)
        self.assertIn("input_errors", result)

if __name__ == "__main__":
    unittest.main()
