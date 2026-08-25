"""Regression checks for terminal Mid Promo -> End Card interval handoff.

Run from the repository root:
    python tools/test_replacement_interval_handoff.py
"""

from __future__ import annotations

import ast
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "api" / "app" / "branding_v09.py"


class FakeHTTPException(Exception):
    def __init__(self, *, status_code, detail):
        self.status_code = status_code
        self.detail = detail
        super().__init__(str(detail))


def load_overlap_resolver():
    module = ast.parse(SOURCE.read_text(encoding="utf-8"))
    names = {
        "_replacement_event_diagnostic",
        "_resolve_replacement_timeline_overlaps",
    }
    nodes = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert len(nodes) == 2, "replacement interval resolver was not found"

    sandbox = {"HTTPException": FakeHTTPException}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), sandbox)
    return sandbox["_resolve_replacement_timeline_overlaps"]


def linked_terminal_events():
    return [
        {
            "kind": "mid_promo_replace",
            "action": {"action_id": "mid-promo-1"},
            "start_seconds": 28.6,
            # Simulates a render-only tail guard / transition expansion.
            "end_seconds": 50.103333,
        },
        {
            "kind": "end_card_replace",
            "action": {
                "action_id": "terminal-end-card-1",
                "source_candidate": {
                    "semantic_mid_promo_action_id": "mid-promo-1"
                },
            },
            # Simulates a visual boundary detector that moves the End Card cut
            # back into the preceding Mid Promo interval.
            "start_seconds": 49.84,
            "end_seconds": 52.167,
        },
    ]


def test_linked_terminal_handoff_is_clamped(resolver):
    events, receipts = resolver(linked_terminal_events(), 30.0)
    assert events[1]["start_seconds"] == events[0]["end_seconds"]
    assert len(receipts) == 1
    assert receipts[0]["previous_action_id"] == "mid-promo-1"
    assert receipts[0]["overlap_seconds_removed"] > 0


def test_unlinked_overlap_stays_a_hard_failure(resolver):
    events = linked_terminal_events()
    events[1]["action"]["source_candidate"] = {}

    try:
        resolver(events, 30.0)
    except FakeHTTPException as exc:
        assert exc.status_code == 422
        assert exc.detail["error"] == "replacement_intervals_overlap"
        assert exc.detail["previous_interval"]["action_id"] == "mid-promo-1"
        assert exc.detail["conflicting_interval"]["action_id"] == "terminal-end-card-1"
    else:
        raise AssertionError("unlinked interval overlap must remain rejected")


def main():
    resolver = load_overlap_resolver()
    test_linked_terminal_handoff_is_clamped(resolver)
    test_unlinked_overlap_stays_a_hard_failure(resolver)
    print("replacement interval handoff regression: OK")


if __name__ == "__main__":
    main()