"""End-to-end regression tests for snapshot and exact-timer input modes."""
import json
import pathlib
import subprocess
import sys
import tempfile

exe = pathlib.Path(sys.argv[1]).resolve()
with tempfile.TemporaryDirectory() as work:
    path = pathlib.Path(work) / "guards.csv"
    path.write_text("3,20\n1,5 # short\n")
    def run(ticks, *flags):
        return subprocess.run([exe, *flags, path, "-"], text=True, input=ticks,
                              capture_output=True, check=False)
    def edges(ticks, *flags):
        result = run(ticks, *flags)
        assert result.returncode == 0, result.stderr
        return [(e["t"], e["id"], e["active"]) for e in map(json.loads, result.stdout.splitlines())]
    ticks = "100,2\n105\t1\n140 0\n"
    assert sorted(edges(ticks)) == [(100,1,True),(105,0,True),(140,0,False),(140,1,False)]
    assert edges(ticks, "--idle") == [(100,1,True),(105,0,True),(110,1,False),(120,0,False)]
    assert edges("9,1\n", "--idle") == [(9,1,True),(14,1,False)]
    assert edges("18446744073709551615,1\n", "--idle") == [(2**64-1,1,True)]
    for invalid in ["9,1\n8,1", "1,-1","1,,2", "1,18446744073709551616", "1", "2,1x"]:
        result = run(invalid)
        assert result.returncode == 2 and "-:" in result.stderr, (invalid, result.stderr)
    path.write_text("4294967296,1")
    assert run("").returncode == 2
print("CLI semantics and malformed-input checks passed")
