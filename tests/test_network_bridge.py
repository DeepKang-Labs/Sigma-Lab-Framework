import os, json, yaml
from network_bridge.network_bridge import NetworkBridge

def test_load_mappings_skywire():
    b = NetworkBridge(".", "./sigma_config_placeholder.yaml", "./network_bridge/mappings_skywire.yaml")
    assert len(b.mappings) >= 1

def test_demo_discovery_and_transform_skywire():
    b = NetworkBridge(".", "./sigma_config_placeholder.yaml", "./network_bridge/mappings_skywire.yaml",
                      network_name="skywire", formula_eval_mode="linear")
    d = b.load_discovery_data()
    ctxs = b.transform_to_sigma_contexts(d)
    assert len(ctxs) >= 1
    for c in ctxs:
        assert 0.0 <= c["short_term_risk"] <= 1.0

def test_load_mappings_fiber():
    b = NetworkBridge(".", "./sigma_config_placeholder.yaml", "./network_bridge/mappings_fiber.yaml",
                      network_name="fiber")
    assert len(b.mappings) >= 1

def test_demo_discovery_and_transform_fiber():
    b = NetworkBridge(".", "./sigma_config_placeholder.yaml", "./network_bridge/mappings_fiber.yaml",
                      network_name="fiber", formula_eval_mode="auto")
    d = b.load_discovery_data()
    ctxs = b.transform_to_sigma_contexts(d)
    assert len(ctxs) >= 1
