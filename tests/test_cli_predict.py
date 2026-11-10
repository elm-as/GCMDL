"""Unit and smoke tests for single-asset prediction and triage CLI."""

import io
import sys
from predict import inspect_asset_risk


def test_inspect_asset_risk_executes_cleanly_for_btc() -> None:
    """Verify that BTC asset inspection prints diagnostic report without errors."""
    captured_stdout = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured_stdout

    try:
        inspect_asset_risk("BTC")
    finally:
        sys.stdout = old_stdout

    output = captured_stdout.getvalue()
    assert "GCMDL ASSET RISK REPORT" in output
    assert "Bitcoin (BTC)" in output
    assert "Turnover Velocity" in output
    assert "Cognitive CoT Diagnostic Rationale" in output
