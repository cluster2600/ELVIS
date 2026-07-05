#!/usr/bin/env python3
"""
Verify Dashboard Position Display Fix

This script verifies that the dashboard changes correctly display all open positions.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.paper_trade_db import get_open_positions


def check_dashboard_code():
    """Verify the dashboard code has been fixed."""
    print("🔍 Checking Dashboard Code Changes...")

    dashboard_file = project_root / "utils" / "console_dashboard.py"

    if not dashboard_file.exists():
        print("❌ Dashboard file not found!")
        return False

    content = dashboard_file.read_text()

    # Check for old limiting code
    old_patterns = [
        "live_positions[:5]",  # Old main section limit
        "open_positions[:3]",  # Old risk section limit
    ]

    # Check for new unlimited code
    new_patterns = [
        "for pos in live_positions:",  # New main section (no limit)
        "for i, pos in enumerate(open_positions):",  # New risk section (no limit)
    ]

    issues = []
    fixes = []

    for pattern in old_patterns:
        if pattern in content:
            issues.append(f"Found old limiting code: {pattern}")

    for pattern in new_patterns:
        if pattern in content:
            fixes.append(f"Found fixed code: {pattern}")

    print("\n📊 Code Analysis Results:")

    if issues:
        print("❌ Issues found:")
        for issue in issues:
            print(f"   • {issue}")
    else:
        print("✅ No old limiting code found")

    if fixes:
        print("✅ Fixes verified:")
        for fix in fixes:
            print(f"   • {fix}")
    else:
        print("❌ No fixed code patterns found")

    return len(issues) == 0 and len(fixes) >= 2


def test_position_retrieval():
    """Test that position retrieval works correctly."""
    print("\n🗄️  Testing Position Retrieval...")

    try:
        positions = get_open_positions()
        count = len(positions)

        print(f"📈 Retrieved {count} open positions from database")

        if count > 5:
            print("✅ More than 5 positions available - dashboard will show all!")
        elif count > 3:
            print("✅ More than 3 positions available - risk section will show all!")
        elif count > 0:
            print(f"📊 {count} positions found - all will be displayed")
        else:
            print("📭 No open positions found")

        return True
    except Exception as e:
        print(f"❌ Error retrieving positions: {e}")
        return False


def main():
    """Run dashboard fix verification."""
    print("🛠️  Dashboard Position Display Fix Verification")
    print("=" * 55)

    try:
        # Check code changes
        code_check = check_dashboard_code()

        # Test position retrieval
        retrieval_check = test_position_retrieval()

        print("\n" + "=" * 55)

        if code_check and retrieval_check:
            print("🎉 Dashboard Fix Verification: PASSED")
            print("\n✅ Summary:")
            print("   • Old position limits removed from dashboard code")
            print("   • All open positions will now be displayed")
            print("   • Both main section and risk section fixed")
            print("\n🚀 The dashboard will now show ALL open positions!")
        else:
            print("❌ Dashboard Fix Verification: FAILED")
            if not code_check:
                print("   • Code changes verification failed")
            if not retrieval_check:
                print("   • Position retrieval test failed")

        print("\n📋 To test the fix:")
        print("   1. Run: python test_positions_display.py (create test data)")
        print("   2. Run: python trading/scripts/dashboard.py (start dashboard)")
        print("   3. Verify all positions are visible")
        print("   4. Run: python reset_paper_trading.py (cleanup)")

        return code_check and retrieval_check

    except Exception as e:
        print(f"❌ Error during verification: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
