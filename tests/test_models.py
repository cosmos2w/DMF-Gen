"""Focused checks for the three public GL-RBF model profiles."""

from __future__ import annotations

import pytest
import torch

from dmf_gen import build_pointcloud_model

MODEL_NAMES = ("GL_rbf", "GL_rbf_ENH", "GL_rbf_ENH_CQ")


def _tiny_config(backbone: str) -> dict[str, object]:
    config: dict[str, object] = {
        "backbone": backbone,
        "coord_dim": 2,
        "hidden_dim": 16,
        "cond_dim": 8,
        "field_embed_dim": 4,
        "latent_dim": 16,
        "num_latents": 4,
        "num_heads": 4,
        "num_latent_blocks": 2,
        "ff_mult": 2,
        "gather_mode": "rbf",
        "gather_topk": 4,
        "neighbor_backend": "torch",
        "rff_features": 8,
        "rff_lengthscale": 0.15,
    }
    if backbone == "GL_rbf_ENH_CQ":
        config.update(
            cq_query_dim=16,
            cq_readout_rank=16,
            cq_readout_heads=4,
            cq_time_embed_dim=16,
        )
    return config


def _batch() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(19)
    batch_size, n_query, n_obs = 2, 9, 6
    coords = torch.rand(batch_size, n_query, 2)
    x1 = torch.randn(batch_size, n_query, 2)
    obs_indices = torch.arange(n_obs).expand(batch_size, -1).clone()
    obs_coords = coords[:, :n_obs].clone()
    obs_field_ids = torch.tensor([[0, 1, 0, 1, 1, 0]]).expand(batch_size, -1).clone()
    obs_mask = torch.tensor([[True, True, True, True, True, False]]).expand(
        batch_size, -1
    ).clone()
    obs_values = x1[
        torch.arange(batch_size).view(-1, 1), obs_indices, obs_field_ids
    ].unsqueeze(-1)
    obs_values[:, -1] = 0.0
    obs_coords[:, -1] = 0.0
    obs_indices[:, -1] = 0
    return x1, coords, obs_coords, obs_values, obs_mask, obs_field_ids, obs_indices


def test_cq_public_aliases_share_the_enhanced_cq_class() -> None:
    from dmf_gen import GL_rbf_CQ as package_alias
    from dmf_gen import GL_rbf_ENH_CQ as package_enhanced
    from dmf_gen.models import GL_rbf_CQ as models_alias
    from dmf_gen.models import GL_rbf_ENH_CQ as models_enhanced

    assert package_alias is package_enhanced
    assert models_alias is models_enhanced
    assert package_alias is models_alias


def test_factory_profiles_preserve_base_and_enhanced_defaults() -> None:
    base = build_pointcloud_model(_tiny_config("GL_rbf"), n_fields=2).model
    enhanced = build_pointcloud_model(_tiny_config("GL_rbf_ENH"), n_fields=2).model

    assert base.sensor_coord_encoding == "raw"
    assert not base.latent_sensor_reinject
    assert not base.query_latent_readout_enabled
    assert base.query_readout_type == "point"
    assert enhanced.sensor_coord_encoding == "fourier"
    assert enhanced.latent_sensor_reinject
    assert enhanced.query_latent_readout_enabled
    assert enhanced.query_readout_type == "coord"

    # Learned width is an explicit option, not an enhanced-only feature.
    assert not base.learnable_rbf_sigma
    assert not hasattr(base, "log_rbf_sigma")
    learned_sigma_config = _tiny_config("GL_rbf")
    learned_sigma_config["learnable_rbf_sigma"] = True
    learned_sigma = build_pointcloud_model(learned_sigma_config, n_fields=2).model
    assert learned_sigma.learnable_rbf_sigma
    assert isinstance(learned_sigma.log_rbf_sigma, torch.nn.Parameter)

    # The coarse GLRES branch follows the gather mode for either profile.
    for backbone in ("GL_rbf", "GL_rbf_ENH"):
        glres_config = _tiny_config(backbone)
        glres_config["gather_mode"] = "topk_rbf_glres"
        glres = build_pointcloud_model(glres_config, n_fields=2).model
        assert hasattr(glres, "coarse_head")
        assert hasattr(glres, "sensor_importance")
    assert not hasattr(base, "coarse_head")


@pytest.mark.parametrize("backbone", MODEL_NAMES)
def test_variants_forward_loss_and_sample(backbone: str) -> None:
    model = build_pointcloud_model(_tiny_config(backbone), n_fields=2)
    x1, coords, obs_coords, obs_values, obs_mask, obs_field_ids, obs_indices = _batch()
    t = torch.tensor([0.2, 0.7])
    x_t = torch.randn_like(x1)

    prediction = model.model(
        t, x_t, coords, obs_coords, obs_values, obs_mask, obs_field_ids
    )
    assert prediction.shape == x1.shape
    assert torch.isfinite(prediction).all()

    loss, metrics = model.training_loss(
        x1=x1,
        coords=coords,
        obs_coords=obs_coords,
        obs_values=obs_values,
        obs_mask=obs_mask,
        obs_field_ids=obs_field_ids,
        obs_indices=obs_indices,
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert metrics["loss"] >= 0.0

    sample = model.sample(
        coords=coords,
        obs_coords=obs_coords,
        obs_values=obs_values,
        obs_mask=obs_mask,
        obs_field_ids=obs_field_ids,
        n_steps=2,
        clamp_indices=obs_indices,
        ode_solver="euler",
        obs_consistency_mode="default_hard",
    )
    assert sample.shape == x1.shape
    assert torch.isfinite(sample).all()
    valid = obs_mask
    bidx = torch.arange(x1.shape[0]).view(-1, 1).expand_as(obs_indices)
    observed = sample[bidx[valid], obs_indices[valid], obs_field_ids[valid]]
    expected = obs_values[..., 0][valid]
    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)


def test_cq_measurement_support_ignores_padded_field_ids() -> None:
    config = _tiny_config("GL_rbf_ENH_CQ")
    config["cq_measurement_support_mode"] = "rbf_value_support"
    with pytest.raises(ValueError, match="requires topk_rbf"):
        build_pointcloud_model(config, n_fields=2)

    config["gather_mode"] = "topk_rbf_glres"
    config["gather_topk"] = 6
    model = build_pointcloud_model(config, n_fields=2)
    x1, coords, obs_coords, obs_values, obs_mask, obs_field_ids, _ = _batch()
    obs_field_ids[:, -1] = -1

    output = model.model(
        torch.tensor([0.2, 0.7]),
        x1,
        coords,
        obs_coords,
        obs_values,
        obs_mask,
        obs_field_ids,
    )
    assert output.shape == x1.shape
    assert torch.isfinite(output).all()
