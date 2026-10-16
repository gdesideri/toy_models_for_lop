"""utils/evaluation.py: unit_health and effective_rank."""
import pytest
import torch

from models.factory import build_model
from utils.evaluation import effective_rank, unit_health


def three_unit_net(activation):
    """mlp_2x3 on 2 inputs whose first layer has one dead unit, one always-on unit and one
    unit on for x0 > 0 (half of a centred cloud)."""
    torch.manual_seed(0)
    net = build_model("mlp_2x3", d_in=2, n_classes=2, activation=activation)
    first = net.body[0]
    with torch.no_grad():
        first.weight.copy_(torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]))
        first.bias.copy_(torch.tensor([-1.0, 1.0, 0.0]))
    return net


def cloud(n=2000):
    return torch.randn(n, 2, generator=torch.Generator().manual_seed(1))


@pytest.mark.parametrize("activation", ["relu", "leaky_relu", "gelu"])
def test_first_layer_counts(activation):
    health = unit_health(three_unit_net(activation), cloud().split(500))
    first = health["layers"][0]
    assert first["units"] == 3 and first["dead"] == 1 and first["rare"] == 1
    assert first["median_rate"] == pytest.approx(0.5, abs=0.05)
    assert first["mean_rate"] == pytest.approx((0 + 1 + 0.5) / 3, abs=0.02)
    assert health["n_images"] == 2000
    assert len(health["layers"]) == 2


def test_rare_counts_units_active_on_under_one_percent():
    net = three_unit_net("relu")
    with torch.no_grad():
        net.body[0].bias[2] = -2.6           # x0 > 2.6: about 0.5% of a standard normal cloud
    first = unit_health(net, cloud().split(500))["layers"][0]
    assert first["dead"] == 1 and first["rare"] == 2


def test_weight_norms_and_effective_rank():
    net = three_unit_net("relu")
    x = cloud()
    health = unit_health(net, x.split(500), probe=x[:1000])
    linear = [m for m in net.modules() if isinstance(m, torch.nn.Linear)]
    assert health["weight_norms"] == pytest.approx([float(m.weight.detach().norm()) for m in linear])
    assert 0 < health["effective_rank"] <= 3


def test_effective_rank_of_known_features():
    assert effective_rank(torch.eye(4)) == pytest.approx(4.0)
    assert effective_rank(torch.ones(10, 1) @ torch.ones(1, 5)) == pytest.approx(1.0)


def test_measuring_changes_nothing():
    net = three_unit_net("relu")
    before = [p.clone() for p in net.parameters()]
    torch.manual_seed(3)
    unit_health(net, cloud().split(500), probe=cloud()[:100])
    after_rng = torch.rand(1)
    torch.manual_seed(3)
    assert torch.equal(after_rng, torch.rand(1))
    assert all(torch.equal(a, b) for a, b in zip(before, net.parameters()))
