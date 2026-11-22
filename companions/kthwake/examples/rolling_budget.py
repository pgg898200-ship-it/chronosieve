"""Signal when posted token charges reach or recover from a rolling budget.

Run `python3 examples/rolling_budget.py` after installing KthWake. Each token
is one unit. Reports with the same second enter one atomic batch. A threshold
of 100 covers the trailing 60 seconds, with charges exactly 60 seconds old
excluded. While the threshold is active, next_expiry() gives the first second
usage would fall below 100 if no more charges arrive.

This covers the threshold-notification part of Kuadrant/limitador issue 545.
The ledger below is an independent check for this small example. A gateway
that reports remaining quota or admits requests of arbitrary cost needs its
own accounting. Cross-replica atomic reservations, Redis persistence and
reservation TTLs belong to that gateway's storage and admission protocol.
"""

from itertools import groupby

from kthwake import Index


LIMIT = 100
WINDOW = 60


def main():
    index = Index([(LIMIT, WINDOW)])
    # A zero charge observes idle time. The first hit starts a fixed window
    # at t=0 for comparison with the rolling count at t=60.
    reports = [
        (0, 1), (59, 35), (59, 25), (60, 20), (60, 20), (61, 60),
        (119, 0), (120, 0), (121, 10**12), (181, 30), (181, 70), (241, 0),
    ]
    ledger = []
    active = False
    observed = []

    def rolling_total(at):
        return sum(charge for stamp, charge in ledger if at - WINDOW < stamp <= at)

    print("second  posted tokens  rolling total  change  next idle recovery")
    for now, group in groupby(reports, key=lambda report: report[0]):
        charge = sum(tokens for _, tokens in group)
        if charge:
            ledger.append((now, charge))
        changes = index.step(now, charge)
        deadline = index.next_expiry()

        # Directly sum the ledger at each possible expiration. This oracle
        # also works for the trillion-token report without expanding it.
        total = rolling_total(now)
        reached = total >= LIMIT
        assert changes == ([(0, reached)] if reached != active else [])
        candidates = (stamp + WINDOW for stamp, _ in ledger if stamp + WINDOW > now)
        expected = next((at for at in candidates if rolling_total(at) < LIMIT), None)
        assert deadline == (expected if reached else None)
        if changes:
            observed.append((now, reached))
        active = reached
        change = ("reach" if reached else "below") if changes else "-"
        print(f"{now:6}  {charge:13,}  {total:13,}  {change:>6}  {deadline or '-'}")

        if now == 60:
            fixed_total = sum(tokens for stamp, tokens in ledger if stamp >= 60)
            assert fixed_total == 40 and total == 100
            print("        The fixed window reset to 40. Rolling usage reached 100.")
        if now == 61:
            assert deadline == 120  # Usage stays exactly at 100 when t=59 expires.
        if now == 181:
            assert not changes and deadline == 241  # Expiry and replacement coincide.

    assert observed == [(60, True), (120, False), (121, True), (241, False)]


if __name__ == "__main__":
    main()
