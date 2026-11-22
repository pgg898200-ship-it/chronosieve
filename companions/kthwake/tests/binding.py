from kthwake import Index
ix = Index(iter([(3, 20), (1, 5), (3, 20)]))
assert ix.step(100, 2) == [(1, True)]
assert sorted(ix.step(105, 1)) == [(0, True), (2, True)]
assert ix.next_expiry() == 110
assert ix.step(110) == [(1, False)]
assert ix.next_expiry() == 120
assert sorted(ix.step(120)) == [(0, False), (2, False)]
assert ix.next_expiry() is None
for f in [lambda: Index([(0, 1)]), lambda: Index([(1, -3)]), lambda: Index([(2**32, 1)]),
          lambda: Index([(1,)]), lambda: ix.step(119), lambda: ix.step(120, 2**64)]:
    try: f()
    except (ValueError, OverflowError): continue
    raise AssertionError("invalid input accepted")
print("Native Python binding passed")
