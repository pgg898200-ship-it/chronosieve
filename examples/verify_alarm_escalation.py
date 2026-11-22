"""Run explicitly after installing the two native dependencies for this example."""

import unittest

from alarm_escalation import incident_updates


class EscalationTest(unittest.TestCase):
    def test_repeated_alarms_and_tied_deadlines(self):
        series = list(enumerate([0.0] * 5 + [3.0] * 11))
        expected = [(6, "watch", True), (12, "escalate", True),
                    (18, "escalate", False), (18, "watch", False)]
        for observations in (series, [*series, (100, 0.0)]):
            with self.subTest(last_t=observations[-1][0]):
                updates = incident_updates(observations)
                actual = sorted((u["t_ms"], u["policy"], u["active"]) for u in updates)
                self.assertEqual(actual, expected)

    def test_simultaneous_detector_alarms(self):
        # Four detector alarms at one time, batched into one policy update.
        observations = [(7, x) for x in [0.0] * 5 + [3.0] * 11]
        updates = incident_updates(observations)
        actual = sorted((u["t_ms"], u["policy"], u["active"]) for u in updates)
        self.assertEqual(actual, [(7, "escalate", True), (7, "watch", True),
                                  (10, "watch", False), (16, "escalate", False)])

    def test_no_alarm(self):
        self.assertEqual(list(incident_updates([(0, 0.0), (1, 0.0)])), [])


if __name__ == "__main__":
    unittest.main()
