import torch

from src.model import ResNet20_PolicyDelta, ResNetCBAM_PolicyDelta, describe_state_dict, load_model


def _check_outputs(model):
    x = torch.zeros(3, 18, 8, 8)
    out = model.eval()(x)
    assert len(out) == 6
    policy, delta_real, delta_scaled, cp_real, cp_scaled, result = out
    assert policy.shape == (3, 20480)
    for t in (delta_real, delta_scaled, cp_real, cp_scaled, result):
        assert t.shape == (3,)
    assert torch.allclose(cp_real, cp_scaled * model.cp_scale)
    assert (result.abs() <= 1).all()


def test_plain_and_cbam_forward_shapes():
    _check_outputs(ResNet20_PolicyDelta(trunk_channels=32, num_blocks=2))
    _check_outputs(ResNetCBAM_PolicyDelta(trunk_channels=32, num_blocks=2))


def test_architecture_is_detected_from_state_dict(tmp_path):
    for cls, arch in ((ResNet20_PolicyDelta, "plain"), (ResNetCBAM_PolicyDelta, "cbam")):
        net = cls(trunk_channels=32, num_blocks=3)
        info = describe_state_dict(net.state_dict())
        assert info == {"arch": arch, "in_planes": 18, "trunk_channels": 32,
                        "num_blocks": 3, "policy_dim": 20480}

        path = tmp_path / f"{arch}.pt"
        torch.save({"epoch": 1, "model": net.state_dict()}, path)  # bundled format
        loaded, loaded_info = load_model(path)
        assert isinstance(loaded, cls)
        assert loaded_info["arch"] == arch
        x = torch.randn(2, 18, 8, 8)
        assert torch.allclose(loaded(x)[0], net.eval()(x)[0])


def test_lfs_pointer_gives_helpful_error(tmp_path):
    path = tmp_path / "pointer.pt"
    path.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 10\n")
    try:
        load_model(path)
    except RuntimeError as e:
        assert "fetch_weights" in str(e)
    else:
        raise AssertionError("expected a RuntimeError for an LFS pointer")
