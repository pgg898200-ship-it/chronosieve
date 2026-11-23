"""Check the exact-timer benchmark's fixture and repetition arguments."""
import pathlib
import subprocess
import sys
import tempfile

exe = pathlib.Path(sys.argv[1]).resolve()
with tempfile.TemporaryDirectory() as tmp:
    guards = pathlib.Path(tmp, "guards")
    ticks = pathlib.Path(tmp, "ticks")
    for path in (guards, ticks):
        for bad in ("5", "2 -1", "1 18446744073709551616", "3 4 5", "3 4x"):
            guards.write_text("1 5\n")
            ticks.write_text("0 1\n")
            with path.open("a") as f:
                f.write(bad)
            result = subprocess.run([exe, guards, ticks, "1"], capture_output=True, text=True)
            assert result.returncode and not result.stdout and f"{path}:2:" in result.stderr, result
    ticks.write_text("0 1\r\n\n18446744073709551615\t1")
    for bad in ("1x", "4294967297", "0", "-1"):
        result = subprocess.run([exe, guards, ticks, bad], capture_output=True)
        assert result.returncode and not result.stdout, result
    result = subprocess.run([exe, guards, ticks, "1"], check=True, capture_output=True, text=True)
    assert len(result.stdout.splitlines()) == 5
