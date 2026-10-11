import copy
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys

import pytest

from engine.core import SigmaAnalyzer
from network_bridge.network_bridge import NetworkBridge
from sigma_lab_v4_2 import SigmaLab, demo_context
from tools.mesh_memory_append import load_or_init_memory
from tools.priority_matrix_from_mappings import build_priority_matrix

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('value', ['invalid', float('nan'), float('inf'), True, -0.1, 1.1])
def test_invalid_risk_never_accepts_and_does_not_mutate(value):
    config, context = demo_context()
    context.short_term_risk = value
    before = copy.deepcopy(context)
    result = SigmaLab(config).diagnose(context)
    assert result['input_errors']
    assert result['status'] == 'invalid_input'
    assert result['verdict'] is None
    assert context.stakeholders == before.stakeholders
    assert context.short_term_risk is value or context.short_term_risk == before.short_term_risk
    json.dumps(result, allow_nan=False)


def test_partial_and_wrapped_configuration_uses_weights():
    _, context = demo_context()
    engine = SigmaLab({'weights': {'values': {'non_harm': 1, 'stability': 0, 'resilience': 0, 'equity': 0}},
                       'thresholds': {'values': {'non_harm_floor': 0.1}}})
    result = engine.diagnose(context)
    assert result['aggregate_score'] == result['scores']['non_harm']
    assert result['audit']['weights']['non_harm'] == 1
    assert result['audit']['thresholds']['veto_irreversibility'] == 0.7


@pytest.mark.parametrize('weight', [-1, float('nan'), float('inf'), True])
def test_invalid_weights_are_rejected(weight):
    with pytest.raises(ValueError):
        SigmaLab({'weights': {'non_harm': weight}})


def test_shared_context_concurrent_diagnostics_are_stable():
    config, context = demo_context()
    before = copy.deepcopy(context)
    engine = SigmaLab(config)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: engine.diagnose(context)['diagnostic'], range(30)))
    assert all(result == results[0] for result in results)
    assert context == before


@pytest.mark.parametrize('field,value', [('avg_uptime', float('nan')), ('success_ratio', 2),
    ('avg_latency', -1), ('node_count', 1.5), ('node_count', True), ('avg_uptime', float('inf'))])
def test_invalid_network_metrics_cannot_be_healthy(field, value):
    metrics = {'node_count': 300, 'avg_uptime': 0.95, 'avg_latency': 25, 'success_ratio': 0.99}
    metrics[field] = value
    with pytest.raises(ValueError):
        SigmaAnalyzer().evaluate(metrics)


@pytest.mark.parametrize('payload', [{}, {'payloads': []}, {'payloads': [{}]}, {'payloads': [None]}])
def test_missing_vitals_are_not_scored(tmp_path, payload):
    path = tmp_path/'vitals.json'
    path.write_text(json.dumps(payload), encoding='utf-8')
    with pytest.raises(ValueError):
        SigmaAnalyzer().evaluate_from_file(path)


@pytest.mark.parametrize('network', ['skywire', 'fiber'])
def test_workflow_validation_command_and_downstream_tools(tmp_path, network):
    output = tmp_path/'validation.json'
    command = [sys.executable, '-m', 'network_bridge.run_network_integrated', '--network', network,
        '--discovery', str(tmp_path), '--mappings', str(ROOT/f'network_bridge/mappings_{network}.yaml'),
        '--config', str(ROOT/'sigma_config_placeholder.yaml'), '--out', str(output), '--validate-only', '--formula-eval', 'auto']
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding='utf-8'))
    assert report['live_transport_established'] is False
    assert report['discovery_source'] == 'demo'
    assert report['contexts'] and report['diagnostics']
    memory = tmp_path/'memory.json'
    result = subprocess.run([sys.executable, '-m', 'tools.mesh_memory_append', '--report', str(output),
        '--memory', str(memory)], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert len(json.loads(memory.read_text(encoding='utf-8'))['runs']) == 1


def test_bad_discovery_is_reported_without_traceback(tmp_path):
    (tmp_path/'decision_mapper.yaml').write_text('decision_points: [null]', encoding='utf-8')
    output = tmp_path/'invalid.json'
    result = subprocess.run([sys.executable, '-m', 'network_bridge.run_network_integrated', '--network', 'skywire',
        '--discovery', str(tmp_path), '--mappings', str(ROOT/'network_bridge/mappings_skywire.yaml'),
        '--config', str(ROOT/'sigma_config_placeholder.yaml'), '--out', str(output), '--validate-only'],
        cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert 'Traceback' not in result.stderr
    assert json.loads(output.read_text())['status'] == 'invalid_input'


def test_context_export_cannot_escape_destination(tmp_path):
    bridge = NetworkBridge(str(tmp_path), str(ROOT/'sigma_config_placeholder.yaml'),
        str(ROOT/'network_bridge/mappings_skywire.yaml'), export_contexts_dir=str(tmp_path/'export'))
    with pytest.raises(ValueError):
        bridge.export_sigma_contexts([{'metadata': {'original_decision_id': '../escaped'}}])
    assert not (tmp_path/'escaped.yaml').exists()


@pytest.mark.parametrize('contents', ['not json', '{}'])
def test_bad_memory_is_not_silently_reset(tmp_path, contents):
    path = tmp_path/'memory.json'
    path.write_text(contents)
    with pytest.raises(ValueError):
        load_or_init_memory(path)
    assert path.read_text() == contents


def test_priority_fallback_is_explicit():
    result = build_priority_matrix({'mappings': []})
    assert result['fallback_used'] is True
    assert result['status'] == 'fallback_no_supported_schema'


def test_weighted_priority_is_not_marked_as_fallback():
    result = build_priority_matrix({'node-a': {'weight': 3}})
    assert result['fallback_used'] is False
    assert result['status'] == 'computed'


@pytest.mark.parametrize('value', [float('nan'), True, -1])
def test_bad_acceptance_threshold_rejected(value):
    with pytest.raises(ValueError):
        SigmaLab({'verdict_acceptance_threshold': value})


def test_analyzer_normalizes_weights_and_rejects_invalid_weights():
    metrics = {'node_count': 300, 'avg_uptime': 0.95, 'avg_latency': 25, 'success_ratio': 0.99}
    analyzer = SigmaAnalyzer(weights={'stability':2, 'latency':0, 'resilience':0, 'equity':0})
    result = analyzer.evaluate(metrics)
    assert result['overall_score'] == round(100*result['component_scores']['stability'],2)
    analyzer.weights['latency'] = float('nan')
    with pytest.raises(ValueError):
        analyzer.evaluate(metrics)


def test_vitals_cli_keeps_local_scope_and_writes_markdown(tmp_path):
    source = tmp_path/'vitals.json'
    source.write_text(json.dumps({'payloads':[{'uptime':0.9,'latency_ms':35,'success_ratio':0.95}]}))
    output = tmp_path/'analysis.json'
    result = subprocess.run([sys.executable,'-m','network_bridge.run_network_integrated','--network','skywire',
        '--input',str(source),'--out',str(output),'--also-md'],cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert result.returncode == 0,result.stderr
    report = json.loads(output.read_text(encoding='utf-8'))
    assert report['scope'] == 'local-vitals-file-analysis'
    assert report['live_transport_established'] is False
    assert 'live transport is not established' in output.with_suffix('.md').read_text(encoding='utf-8')
