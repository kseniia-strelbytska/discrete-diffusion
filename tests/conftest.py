import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--seq-length", type=int, default=8,
        help="Sequence length L for grammar oracle tests (default: 8)",
    )
    parser.addoption(
        "--n-samples", type=int, default=10000,
        help="Number of random (seq, mask) pairs to test per grammar (default: 10000)",
    )


@pytest.fixture(scope='session')
def seq_length(request):
    return request.config.getoption("--seq-length")


@pytest.fixture(scope='session')
def n_samples(request):
    return request.config.getoption("--n-samples")
