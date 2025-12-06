from normalization.canon import normalize_pretty

def test_normalize_implications():
    assert normalize_pretty("p -> q -> r") == "p -> (q -> r)"
    assert normalize_pretty("(p -> q) -> r") == "(p -> q) -> r"
