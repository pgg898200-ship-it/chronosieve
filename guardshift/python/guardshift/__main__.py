"""Read one index definition followed by atomic JSON array patches on stdin."""
import json
import sys

from . import Index


def main():
    """Print one flushed JSON array of transitions per valid patch."""
    line_no = 1
    try:
        first = json.loads(next(sys.stdin))
        index = Index(first["initial"], first["guards"])
        for line_no, line in enumerate(sys.stdin, 2):
            changes = index.apply(json.loads(line))
            print(json.dumps(changes, separators=(",", ":")), flush=True)
    except (StopIteration, ValueError, TypeError, IndexError, KeyError, OverflowError) as exc:
        print(f"guardshift: invalid input at line {line_no}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
