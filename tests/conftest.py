def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a CUDA device; run by verify_equivalence.sh --gpu")
