import pytest
import importlib


def pytest_addoption(parser):
    parser.addoption('--run-llm', action='store_true', help='Run model integration tests; may download weights.')


def pytest_collection_modifyitems(config, items):
    if config.getoption('--run-llm'):
        return
    skip = pytest.mark.skip(reason='Model integration is opt-in: install the LLM stack and pass --run-llm.')
    for item in items:
        if 'llm' in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def llm_agent(tmp_path, monkeypatch):
    for name in ('CONFIGS', 'STATE', 'REPORTS', 'OUTPUTS'):
        monkeypatch.setenv(f'SIGMA_{name}_DIR', str(tmp_path / name.lower()))
    monkeypatch.setenv('SIGMA_ENABLE_JUDGE', '0')
    monkeypatch.setenv('SIGMA_ENABLE_RAG', '0')
    import sigma_llm_complete
    sl = importlib.reload(sigma_llm_complete)
    return sl, sl.SigmaLLM()
