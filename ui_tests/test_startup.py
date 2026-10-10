import importlib.util
import socket
import sys
from pathlib import Path
from urllib.request import urlopen


def test_interface_serves_without_loading_model(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root))
    for name in ('CONFIGS', 'STATE', 'REPORTS', 'OUTPUTS'):
        monkeypatch.setenv(f'SIGMA_{name}_DIR', str(tmp_path / name.lower()))
    monkeypatch.setenv('GRADIO_ANALYTICS_ENABLED', 'False')
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    spec = importlib.util.spec_from_file_location('sigma_ui_under_test', root / 'app.py')
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    assert app._agent is None
    assert app.SigmaLLM is None
    assert 'torch' not in sys.modules
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    try:
        app.demo.launch(server_name='127.0.0.1', server_port=port,
                        prevent_thread_lock=True, quiet=True, share=False)
        with urlopen(f'http://127.0.0.1:{port}/', timeout=10) as response:
            assert response.status == 200
            assert b'Sigma-LLM' in response.read()
        monkeypatch.setattr(app, 'import_sigma_llm', lambda: (_ for _ in ()).throw(ImportError('missing optional dependency')))
        assert app.chat_fn('Hello', [], 0.9, 0.9).startswith('Model unavailable:')
        assert app._agent is None
    finally:
        app.demo.close()
