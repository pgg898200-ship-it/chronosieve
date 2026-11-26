"""A benchmark must reject edits that preserve its expected transition count."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

script = Path(__file__).resolve().parents[1] / "replay" / "measure.py"
with tempfile.TemporaryDirectory() as work:
    inputs = Path(work)
    fixtures = {"ticks.txt": b"0 1\n1 1\n", "shared.guards": b"1 5\n1 6\n"}
    manifest = {
        "trace": "fixture",
        "ticks_sha256": hashlib.sha256(fixtures["ticks.txt"]).hexdigest(),
        "scenarios": {"shared": {"guards_sha256": hashlib.sha256(fixtures["shared.guards"]).hexdigest()}},
    }
    (inputs / "oracle.json").write_text(json.dumps(manifest))
    output = inputs / "results.json"
    for name, changed in (("ticks.txt", b"1 1\n2 1\n"), ("shared.guards", b"1 6\n1 5\n")):
        for filename, data in fixtures.items():
            (inputs / filename).write_bytes(data)
        (inputs / name).write_bytes(changed)
        result = subprocess.run([sys.executable, script, "--inputs", inputs, "--out", output],
                                capture_output=True, text=True)
        assert result.returncode and f"{name}: SHA-256 mismatch" in result.stderr, result
        assert not output.exists()
