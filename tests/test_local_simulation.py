import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from engine.context_diffusion import laplacian_from_graph
from engine.integrators import rk4_step
from engine.invariants import check_cfl, check_petit_gain

ROOT = Path(__file__).resolve().parents[1]


def test_laplacian_matches_known_graph_and_ignores_self_loops():
    weights = np.array([[1,2,0],[2,1,3],[0,3,1]],dtype=float)
    expected = np.array([[2,-2,0],[-2,5,-3],[0,-3,3]],dtype=float)
    np.testing.assert_array_equal(laplacian_from_graph(weights),expected)


@pytest.mark.parametrize('matrix', [[[1,2]], [[1,-1],[-1,1]], [[1,2],[0,1]], [[float('nan')]]])
def test_invalid_graph_rejected(matrix):
    with pytest.raises(ValueError):
        laplacian_from_graph(matrix)


@pytest.mark.parametrize('value', [-1,float('nan'),float('inf'),True])
def test_invalid_invariant_parameters_do_not_pass(value):
    assert not check_cfl(value,1,0.2)
    assert not check_cfl(0.1,value,0.2)
    assert not check_petit_gain(value,0.1,0.01)


def test_rk4_decay_has_expected_accuracy():
    state = np.array([1.0])
    for i in range(10):
        state = rk4_step(lambda t,y:-y,state,i/10,0.1)
    assert abs(state[0]-np.exp(-1)) < 4e-7


def test_complete_simulation_is_reproducible_without_external_artifacts(tmp_path):
    for folder, filename in [('configs','sigma_params.json'),('policy','safety_policy.yaml')]:
        (tmp_path/folder).mkdir()
        shutil.copy2(ROOT/folder/filename,tmp_path/folder/filename)
    env=dict(os.environ,PYTHONPATH=str(ROOT))
    reports=[]
    for _ in range(2):
        result=subprocess.run([sys.executable,'-m','pipelines.run_sigma','--seed','7'],cwd=tmp_path,
                              env=env,capture_output=True,text=True,timeout=30)
        assert result.returncode==0,result.stderr
        reports.append(json.loads((tmp_path/'state/last_metrics.json').read_text(encoding='utf-8')))
    assert reports[0]==reports[1]
    assert reports[0]['live_transport_established'] is False
    assert all(np.isfinite(reports[0]['final_C']))
