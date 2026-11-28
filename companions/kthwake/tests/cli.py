"""End-to-end regression tests for snapshot and exact-timer input modes."""
import json
import pathlib
import queue
import subprocess
import sys
import tempfile
import threading

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
    # The producer keeps its pipe open and waits for a change before its next tick.
    for flags in ([], ["--idle"]):
        with subprocess.Popen([exe, *flags, path, "-"], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, text=True) as process:
            received = queue.Queue()
            def read_output():
                for line in process.stdout:
                    received.put(json.loads(line))
            reader = threading.Thread(target=read_output)
            reader.start()
            try:
                for tick, expected in [("100,2", (100, 1, True)), ("105,1", (105, 0, True))]:
                    process.stdin.write(tick + "\n")
                    process.stdin.flush()
                    edge = received.get(timeout=5)
                    assert (edge["t"], edge["id"], edge["active"]) == expected
            finally:
                process.stdin.close()
                reader.join(timeout=5)
                if reader.is_alive():
                    process.kill()
                    reader.join()
            assert process.wait(timeout=5) == 0
            if flags:
                for t, rule in [(110, 1), (120, 0)]:
                    assert received.get_nowait() == {"t": t, "id": rule, "active": False}
            assert received.empty()
    for invalid in ["9,1\n8,1", "1,-1","1,,2", "1,18446744073709551616", "1", "2,1x"]:
        result = run(invalid)
        assert result.returncode == 2 and "-:" in result.stderr, (invalid, result.stderr)
    path.write_text("4294967296,1")
    assert run("").returncode == 2
print("CLI semantics and malformed-input checks passed")
