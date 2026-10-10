import pytest


def pytest_addoption(parser):
    parser.addoption('--run-llm', action='store_true', help='Run model integration tests; may download weights.')


def pytest_collection_modifyitems(config, items):
    if config.getoption('--run-llm'):
        return
    skip = pytest.mark.skip(reason='Model integration is opt-in: install the LLM stack and pass --run-llm.')
    for item in items:
        if 'llm' in item.keywords:
            item.add_marker(skip)
