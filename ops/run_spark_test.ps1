$env:DATABASE_URL = 'postgresql://first_organism_admin:f1rst_0rg4n1sm_l0c4l_s3cur3_k3y!@127.0.0.1:5432/mathledger_first_organism'
$env:REDIS_URL = 'redis://:r3d1s_f1rst_0rg_s3cur3!@127.0.0.1:6380/0'
$env:LEDGER_API_KEY = 'sk_first_organism_test_key_v1_2025'
$env:FIRST_ORGANISM_TESTS = 'true'
$env:FIRST_ORGANISM_STRICT_MODE = 'true'
$env:FIRST_ORGANISM_REQUIRE_CLEAN_DB = 'true'
$env:RUNTIME_ENV = 'test_hardened'
uv run pytest tests/integration/test_first_organism.py::test_first_organism_closed_loop -v -s

