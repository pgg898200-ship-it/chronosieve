"""Emit the pinned Azure code inference replay. Requires only Python 3."""
import argparse
import csv
import hashlib
import io
import itertools
import sys
import urllib.request
from collections import deque

REVISION = "5d7424495ca07a30c89d30a2a534ac0d0d3a5e3e"
URL = (
    "https://raw.githubusercontent.com/Azure/AzurePublicDataset/"
    + REVISION + "/data/AzureLLMInferenceTrace_code.csv"
)
REQUESTS = 4096
WINDOW = 64
SOURCE_PREFIX_SHA256 = "00eac94f2e9cfadd042700d3fff36964013c686059bc171e82afa537ebaf5dda"


def read_prefix(stream):
    lines = [line.rstrip(b"\r\n") + b"\n" for line in itertools.islice(stream, REQUESTS + 1)]
    prefix = b"".join(lines)
    if hashlib.sha256(prefix).hexdigest() != SOURCE_PREFIX_SHA256:
        raise ValueError("The first 4096 data rows differ from the pinned source")
    return csv.DictReader(io.StringIO(prefix.decode("ascii")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="local copy of AzureLLMInferenceTrace_code.csv")
    args = parser.parse_args()
    source = open(args.source, "rb") if args.source else urllib.request.urlopen(URL, timeout=30)
    with source:
        rows = read_prefix(source)
    window = deque()
    context_total = generated_total = 0
    output = csv.writer(sys.stdout, lineterminator="\n")
    output.writerow(["field", "value"])
    for row in rows:
        context, generated = int(row["ContextTokens"]), int(row["GeneratedTokens"])
        if len(window) == WINDOW:
            old_context, old_generated = window.popleft()
            context_total -= old_context
            generated_total -= old_generated
        window.append((context, generated))
        context_total += context
        generated_total += generated
        output.writerow([0, context_total])
        output.writerow([1, generated_total])


if __name__ == "__main__":
    main()
