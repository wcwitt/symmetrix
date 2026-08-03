from math import sqrt

import pytest

try:
    import torch

    torch.serialization.add_safe_globals([slice])
    from mace.tools.scripts_utils import remove_pt_head
except ImportError as exc:
    pytest.skip(
        f"mace-field tensor test dependencies are not available: {exc}",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def macefield_modules(macefield_model_path):
    model = torch.load(
        macefield_model_path, map_location=torch.device("cpu"), weights_only=False
    ).to(torch.float64)
    if hasattr(model, "heads") and len(model.heads) != 1:
        torch.set_default_dtype(next(model.parameters()).dtype)
        model = remove_pt_head(model, "mp-dielectric")
    model.eval()

    return model.field_feats[0], model.field_linear[0]


def _frozen_inputs():
    generator = torch.Generator(device="cpu").manual_seed(20260710)
    h1_pre = torch.randn(7, 512, dtype=torch.float64, generator=generator)
    electric_field = torch.randn(7, 3, dtype=torch.float64, generator=generator)
    return h1_pre, electric_field


def _weight_views(module):
    offset = 0
    views = []
    for instruction in module.instructions:
        numel = 1
        for size in instruction.path_shape:
            numel *= size
        views.append(
            module.weight.detach()
            .narrow(0, offset, numel)
            .reshape(instruction.path_shape)
        )
        offset += numel
    assert offset == module.weight.numel()
    return views


def _compact_field_coupling(field_feats, field_linear):
    return {
        "field_feats": {
            "irreps_in1": str(field_feats.irreps_in1),
            "irreps_in2": str(field_feats.irreps_in2),
            "irreps_out": str(field_feats.irreps_out),
            "output_mask": field_feats.output_mask.detach().clone(),
            "instructions": [
                {
                    "i_in1": instruction.i_in1,
                    "i_in2": instruction.i_in2,
                    "i_out": instruction.i_out,
                    "connection_mode": instruction.connection_mode,
                    "path_shape": instruction.path_shape,
                    "path_weight": instruction.path_weight,
                    "weight": weight.clone(),
                }
                for instruction, weight in zip(
                    field_feats.instructions, _weight_views(field_feats)
                )
            ],
        },
        "field_linear": {
            "irreps_in": str(field_linear.irreps_in),
            "irreps_out": str(field_linear.irreps_out),
            "output_mask": field_linear.output_mask.detach().clone(),
            "bias": field_linear.bias.detach().clone(),
            "instructions": [
                {
                    "i_in": instruction.i_in,
                    "i_out": instruction.i_out,
                    "path_shape": instruction.path_shape,
                    "path_weight": instruction.path_weight,
                    "weight": weight.clone(),
                }
                for instruction, weight in zip(
                    field_linear.instructions, _weight_views(field_linear)
                )
            ],
        },
    }


def _standalone_field_feats(coupling, h1_pre, electric_field):
    field_feats = coupling["field_feats"]
    assert field_feats["irreps_in1"] == "128x0e+128x1o"
    assert field_feats["irreps_in2"] == "1x1o"
    assert field_feats["irreps_out"] == "128x0e+128x1o"

    scalar_in = h1_pre[..., :128]
    vector_in = h1_pre[..., 128:].reshape(*h1_pre.shape[:-1], 128, 3)
    field = electric_field.reshape(*electric_field.shape[:-1], 1, 3)

    scalar_out = torch.zeros_like(scalar_in)
    vector_out = torch.zeros_like(vector_in)

    for instruction in field_feats["instructions"]:
        weight = instruction["weight"].to(dtype=h1_pre.dtype, device=h1_pre.device)
        assert instruction["connection_mode"] == "uvw"
        assert instruction["path_shape"] == (128, 1, 128)

        if (instruction["i_in1"], instruction["i_in2"], instruction["i_out"]) == (
            0,
            0,
            1,
        ):
            vector_out += (
                instruction["path_weight"]
                * torch.einsum("uvw,...u,...vj->...wj", weight, scalar_in, field)
                / sqrt(3.0)
            )
        elif (instruction["i_in1"], instruction["i_in2"], instruction["i_out"]) == (
            1,
            0,
            0,
        ):
            scalar_out += (
                instruction["path_weight"]
                * torch.einsum("uvw,...ui,...vi->...w", weight, vector_in, field)
                / sqrt(3.0)
            )
        else:
            raise AssertionError(f"Unsupported field_feats instruction: {instruction}")

    output = torch.cat(
        [scalar_out, vector_out.reshape(*h1_pre.shape[:-1], 384)], dim=-1
    )
    return output * field_feats["output_mask"].to(
        dtype=output.dtype, device=output.device
    )


def _standalone_field_linear(coupling, delta_feats):
    field_linear = coupling["field_linear"]
    assert field_linear["irreps_in"] == "128x0e+128x1o"
    assert field_linear["irreps_out"] == "128x0e+128x1o"
    assert field_linear["bias"].numel() == 0

    scalar_in = delta_feats[..., :128]
    vector_in = delta_feats[..., 128:].reshape(*delta_feats.shape[:-1], 128, 3)

    scalar_out = torch.zeros_like(scalar_in)
    vector_out = torch.zeros_like(vector_in)

    for instruction in field_linear["instructions"]:
        weight = instruction["weight"].to(
            dtype=delta_feats.dtype, device=delta_feats.device
        )
        assert instruction["path_shape"] == (128, 128)

        if (instruction["i_in"], instruction["i_out"]) == (0, 0):
            scalar_out += instruction["path_weight"] * torch.einsum(
                "uw,...u->...w", weight, scalar_in
            )
        elif (instruction["i_in"], instruction["i_out"]) == (1, 1):
            vector_out += instruction["path_weight"] * torch.einsum(
                "uw,...ui->...wi", weight, vector_in
            )
        else:
            raise AssertionError(f"Unsupported field_linear instruction: {instruction}")

    output = torch.cat(
        [scalar_out, vector_out.reshape(*delta_feats.shape[:-1], 384)], dim=-1
    )
    return output * field_linear["output_mask"].to(
        dtype=output.dtype, device=output.device
    )


def _standalone_field_transform(coupling, h1_pre, electric_field):
    delta_feats = _standalone_field_feats(coupling, h1_pre, electric_field)
    return h1_pre - _standalone_field_linear(coupling, delta_feats)


def test_checkpoint_field_coupling_uses_supported_hidden_layout(macefield_modules):
    field_feats, field_linear = macefield_modules

    assert str(field_feats.irreps_in1) == "128x0e+128x1o"
    assert str(field_feats.irreps_in2) == "1x1o"
    assert str(field_feats.irreps_out) == "128x0e+128x1o"
    assert str(field_linear.irreps_in) == "128x0e+128x1o"
    assert str(field_linear.irreps_out) == "128x0e+128x1o"

    assert [instruction.path_shape for instruction in field_feats.instructions] == [
        (128, 1, 128),
        (128, 1, 128),
    ]
    assert [instruction.path_shape for instruction in field_linear.instructions] == [
        (128, 128),
        (128, 128),
    ]
    assert torch.all(field_feats.output_mask == 1)
    assert torch.all(field_linear.output_mask == 1)


def test_standalone_field_feats_matches_pytorch_macefield(macefield_modules):
    field_feats, _ = macefield_modules
    coupling = _compact_field_coupling(*macefield_modules)
    h1_pre, electric_field = _frozen_inputs()

    expected = field_feats(h1_pre, electric_field)
    actual = _standalone_field_feats(coupling, h1_pre, electric_field)

    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_standalone_field_transform_matches_pytorch_macefield(macefield_modules):
    field_feats, field_linear = macefield_modules
    coupling = _compact_field_coupling(field_feats, field_linear)
    h1_pre, electric_field = _frozen_inputs()

    expected = h1_pre - field_linear(field_feats(h1_pre, electric_field))
    actual = _standalone_field_transform(coupling, h1_pre, electric_field)

    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)
