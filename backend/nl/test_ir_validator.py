import unittest
import json
import hashlib
from backend.nl.ir_validator import validate_ir, attach_nl_evidence
from backend.nl import nl_ir_failure_codes as codes

# --- Test Fixtures ---

VALID_IR = {
    "statement_id": "NL_TEST_001",
    "source_nl": "The quick brown fox jumps over the lazy dog.",
    "logical_form": {},
    "grounding_context": {
        "fox-01": {"nl_reference": ["The quick brown fox"], "schema_ref": "animal:Canidae"},
        "dog-01": {"nl_reference": ["the lazy dog"], "schema_ref": "animal:Canidae"}
    },
    "confidence_score": 0.95,
    "disambiguation_notes": []
}

# Same content as VALID_IR, but with a different key order.
VALID_IR_REORDERED = {
    "source_nl": "The quick brown fox jumps over the lazy dog.",
    "disambiguation_notes": [],
    "statement_id": "NL_TEST_001",
    "confidence_score": 0.95,
    "logical_form": {},
    "grounding_context": {
        "dog-01": {"schema_ref": "animal:Canidae", "nl_reference": ["the lazy dog"]},
        "fox-01": {"nl_reference": ["The quick brown fox"], "schema_ref": "animal:Canidae"}
    }
}


AMBIGUOUS_IR_FLAG = {
    "statement_id": "NL_TEST_004",
    "source_nl": "He saw her.",
    "logical_form": {},
    "grounding_context": {
        "e1": {"nl_reference": ["He"], "schema_ref": "Person"},
        "e2": {"nl_reference": ["her"], "schema_ref": "Person"},
    },
    "confidence_score": 0.60,
    "disambiguation_notes": [{"text": "He", "interpretation": "Ambiguous pronoun.", "resolution": "..."}]
}

ONTOLOGY_MISMATCH_IR = {
    "statement_id": "NL_TEST_005",
    "source_nl": "The wumpus lives in the cave.",
    "logical_form": {},
    "grounding_context": {
        "w-01": {"nl_reference": ["The wumpus"], "schema_ref": "mythical:Wumpus"},
    },
    "confidence_score": 0.99,
    "disambiguation_notes": []
}


class TestIRValidator(unittest.TestCase):

    def test_ambiguity_flag_fails_semantic_validation(self):
        """An IR with disambiguation_notes should fail semantic validation."""
        is_valid, errors = validate_ir(AMBIGUOUS_IR_FLAG)
        self.assertFalse(is_valid)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]['code'], codes.AMBIGUITY_DETECTED)
        self.assertEqual(errors[0]['path'], 'disambiguation_notes')

    def test_ontology_mismatch_fails_semantic_validation(self):
        """An IR with an unknown schema_ref should fail semantic validation."""
        is_valid, errors = validate_ir(ONTOLOGY_MISMATCH_IR)
        self.assertFalse(is_valid)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]['code'], codes.ONTOLOGY_MISMATCH)
        self.assertEqual(errors[0]['path'], 'grounding_context.w-01.schema_ref')


class TestEvidenceAttachment(unittest.TestCase):

    def test_attachment_is_non_mutating(self):
        """The attach_nl_evidence function should not modify the original evidence pack."""
        original_pack = {"id": "EV-123", "data": {"existing": True}}
        original_pack_copy = original_pack.copy() # Shallow copy is enough to check for top-level mutation
        
        attach_nl_evidence(original_pack, VALID_IR)
        
        self.assertEqual(original_pack, original_pack_copy, "Original evidence pack was mutated.")

    def test_governance_block_for_valid_ir(self):
        """Test the structure of the governance block for a valid IR."""
        evidence_pack = {"id": "EV-123"}
        new_pack = attach_nl_evidence(evidence_pack, VALID_IR)

        self.assertIn('governance', new_pack)
        self.assertIn('nl_ir', new_pack['governance'])
        
        gov_block = new_pack['governance']['nl_ir']
        self.assertEqual(gov_block['schema_version'], "1.0.0")
        self.assertEqual(len(gov_block['payload_hash']), 64) # SHA256
        self.assertEqual(gov_block['findings'], [])

    def test_governance_block_for_invalid_ir(self):
        """Test the findings in the governance block for an invalid IR."""
        evidence_pack = {"id": "EV-456"}
        new_pack = attach_nl_evidence(evidence_pack, AMBIGUOUS_IR_FLAG)

        gov_block = new_pack['governance']['nl_ir']
        self.assertEqual(len(gov_block['findings']), 1)
        self.assertEqual(gov_block['findings'][0]['code'], codes.AMBIGUITY_DETECTED)

    def test_hash_is_deterministic_across_key_order(self):
        """A hash should be identical for the same content regardless of key order."""
        pack1 = attach_nl_evidence({}, VALID_IR)
        pack2 = attach_nl_evidence({}, VALID_IR_REORDERED)

        hash1 = pack1['governance']['nl_ir']['payload_hash']
        hash2 = pack2['governance']['nl_ir']['payload_hash']

        self.assertEqual(hash1, hash2, "Hashes do not match for reordered keys.")
        self.assertNotEqual(json.dumps(VALID_IR), json.dumps(VALID_IR_REORDERED), "Test fixtures are not properly reordered.")

    def test_output_is_json_serializable_and_deterministic(self):
        """The output pack and hash should be deterministic and serializable."""
        pack1 = attach_nl_evidence({}, VALID_IR)
        pack2 = attach_nl_evidence({}, VALID_IR)

        # Hashes should be identical for identical inputs
        self.assertEqual(pack1['governance']['nl_ir']['payload_hash'], pack2['governance']['nl_ir']['payload_hash'])

        # The entire structure should be serializable
        try:
            serialized = json.dumps(pack1)
            deserialized = json.loads(serialized)
            self.assertEqual(pack1, deserialized)
        except TypeError:
            self.fail("attach_nl_evidence output is not JSON serializable.")


if __name__ == '__main__':
    unittest.main()
