import numpy as np

from policy_improvement_lab.mdp import make_branching_mdp
from policy_improvement_lab.operators import ALL_METHODS
from policy_improvement_lab.training import RunConfig, train


def test_smoke_and_reproducibility():
    mdp = make_branching_mdp()
    cfg = RunConfig(steps=40, eval_every=20)
    for method in ALL_METHODS:
        checkpoints, diagnostics = train(mdp, method, 7, cfg)
        assert len(checkpoints) == 3
        assert all(np.isfinite(r["expected_return"]) for r in checkpoints)
        if method == "ac_td0":
            assert not diagnostics
        else:
            assert len(diagnostics) == 12
        checkpoints_2, diagnostics_2 = train(mdp, method, 7, cfg)
        assert checkpoints == checkpoints_2
        assert diagnostics == diagnostics_2
