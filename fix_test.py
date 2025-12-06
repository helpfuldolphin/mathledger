#!/usr/bin/env python
"""Fix test_exporter_v1.py fixture to use valid V1 schema."""

with open('tests/qa/test_exporter_v1.py', 'r') as f:
    content = f.read()

# Replace the invalid fixture
old_fixture = '''        rec = {"system":"fol","mode":"baseline","method":"fol-baseline","seed":"1",
               "inserted_proofs":1,"wall_minutes":0.1,"block_no":1,"merkle":"0"*64}'''

new_fixture = '''        # Valid V1 statement record (matches exporter required_fields)
        rec = {"id": "stmt-1", "theory_id": "pl", "hash": "a"*64,
               "content_norm": "p -> p", "is_axiom": False}'''

updated = content.replace(old_fixture, new_fixture)

with open('tests/qa/test_exporter_v1.py', 'w') as f:
    f.write(updated)

print("✓ Fixed test_dry_run_valid_v1_ok fixture")
