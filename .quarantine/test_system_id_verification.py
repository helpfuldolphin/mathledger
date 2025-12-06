#!/usr/bin/env python3
"""
Test script to verify that the derive function properly filters by system_id.
This script creates statements for two different systems and verifies that
derivations only occur within the correct system.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.axiom_engine.model import Base, Statement
from backend.axiom_engine.derive import derive
from datetime import datetime
import hashlib

def test_system_id_filtering():
    """Test that derive function only works within the specified system_id."""

    # Create an in-memory SQLite database
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    session = Session()

    # Create statements for system 1
    system1_statements = [
        Statement(
            system_id="system1",
            normalized_text="p",
            hash=hashlib.sha256("p".encode()).hexdigest(),
            created_at=datetime.now()
        ),
        Statement(
            system_id="system1",
            normalized_text="p -> q",
            hash=hashlib.sha256("p -> q".encode()).hexdigest(),
            created_at=datetime.now()
        )
    ]

    # Create statements for system 2
    system2_statements = [
        Statement(
            system_id="system2",
            normalized_text="r",
            hash=hashlib.sha256("r".encode()).hexdigest(),
            created_at=datetime.now()
        ),
        Statement(
            system_id="system2",
            normalized_text="r -> s",
            hash=hashlib.sha256("r -> s".encode()).hexdigest(),
            created_at=datetime.now()
        )
    ]

    # Add all statements to database
    session.add_all(system1_statements + system2_statements)
    session.commit()

    print("Initial state:")
    print("System 1 statements:", [s.normalized_text for s in session.query(Statement).filter(Statement.system_id == "system1").all()])
    print("System 2 statements:", [s.normalized_text for s in session.query(Statement).filter(Statement.system_id == "system2").all()])

    # Run derive for system 1 only
    derive(session, "system1", steps=1, breadth_cap=10, total_cap=100)

    print("\nAfter deriving for system1:")
    system1_after = session.query(Statement).filter(Statement.system_id == "system1").all()
    system2_after = session.query(Statement).filter(Statement.system_id == "system2").all()

    print("System 1 statements:", [s.normalized_text for s in system1_after])
    print("System 2 statements:", [s.normalized_text for s in system2_after])

    # Verify that system 1 got new derived statements
    system1_derived = [s for s in system1_after if s.derivation_rule == "MP"]
    print(f"System 1 derived statements: {[s.normalized_text for s in system1_derived]}")

    # Verify that system 2 was not affected
    system2_derived = [s for s in system2_after if s.derivation_rule == "MP"]
    print(f"System 2 derived statements: {[s.normalized_text for s in system2_derived]}")

    # Assertions
    assert len(system1_derived) > 0, "System 1 should have derived statements"
    assert len(system2_derived) == 0, "System 2 should not have derived statements"
    assert "q" in [s.normalized_text for s in system1_derived], "System 1 should have derived 'q'"
    assert "s" not in [s.normalized_text for s in system1_derived], "System 1 should not have derived 's'"

    print("\n✅ All assertions passed! System ID filtering is working correctly.")

    session.close()

if __name__ == "__main__":
    test_system_id_filtering()
