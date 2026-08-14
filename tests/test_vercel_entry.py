import importlib.util
from pathlib import Path
import unittest


class TestVercelEntry(unittest.TestCase):
    def test_vercel_entrypoint_exists(self):
        entry_file = Path(__file__).resolve().parents[1] / "api" / "index.py"
        self.assertTrue(entry_file.exists(), "Missing Vercel entrypoint at api/index.py")

        spec = importlib.util.spec_from_file_location("vercel_entry", entry_file)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)

        self.assertTrue(hasattr(module, "app"), "Expected a FastAPI app exported as 'app'")


if __name__ == "__main__":
    unittest.main()
