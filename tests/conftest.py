import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--seq-length", type=int, default=8,
        help="Sequence length L for grammar oracle tests (default: 8)",
    )


@pytest.fixture(scope='session')
def seq_length(request):
    return request.config.getoption("--seq-length")
