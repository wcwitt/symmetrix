import json

import numpy as np
import pytest

try:
    from symmetrix.extract_mace_data import extract_mace_data
except ImportError as exc:
    extract_mace_data = None
    extract_mace_data_import_error = exc
else:
    extract_mace_data_import_error = None


@pytest.mark.skipif(
    extract_mace_data is None,
    reason=f"extract_mace_data is not available: {extract_mace_data_import_error}",
)
def test_macefield_extractor_includes_field_schema(macefield_model_path):
    data = extract_mace_data(
        macefield_model_path,
        species=[7, 13],
        head="mp-dielectric",
        num_spline_points=8,
    )

    assert data["model_type"] == "MACEField"
    assert data["has_field_coupling"] is True
    assert len(data["field_couplings"]) == 1

    coupling = data["field_couplings"][0]
    assert coupling["field_feats_irreps_in1"] == "128x0e+128x1o"
    assert coupling["field_feats_irreps_in2"] == "1x1o"
    assert coupling["field_feats_irreps_out"] == "128x0e+128x1o"
    assert coupling["field_linear_irreps_in"] == "128x0e+128x1o"
    assert coupling["field_linear_irreps_out"] == "128x0e+128x1o"

    assert len(coupling["field_feats_weight"]) == 32768
    assert len(coupling["field_feats_output_mask"]) == 512
    assert len(coupling["field_linear_weight"]) == 32768
    assert coupling["field_linear_bias"] == []
    assert len(coupling["field_linear_output_mask"]) == 512

    assert len(data["H1_product_weights"]) == 32768
    assert len(data["H1_linear_up_weights"]) == 32768

    assert data["symmetrix_format_version"] == 2
    assert data["radial_representation"] == "compact"
    assert "radial_spline_values_0" not in data
    assert "radial_spline_values_1" not in data
    assert "A0_spline_values" not in data
    assert "A1_spline_values" not in data

    radial = data["compact_radial"]
    assert radial["num_spline_points"] == 8
    assert radial["basis"]["type"] == "bessel"
    assert radial["cutoff"]["type"] == "polynomial"
    assert radial["distance_transform"]["type"] == "agnesi"
    assert len(radial["distance_transform"]["covalent_radii"]) == 2
    assert radial["networks"]["R0"]["shape"] == [10, 64, 64, 64, 512]
    assert radial["networks"]["R1"]["shape"] == [10, 64, 64, 64, 1280]
    assert radial["networks"]["A0"]["postprocess"] == "tanh-square"
    assert radial["networks"]["A1"]["postprocess"] == "tanh-square"


@pytest.mark.skipif(
    extract_mace_data is None,
    reason=f"extract_mace_data is not available: {extract_mace_data_import_error}",
)
def test_macefield_extractor_rejects_unsupported_lmax(
    macefield_model_path,
    monkeypatch,
):
    from e3nn.o3 import Irreps
    from mace.tools.scripts_utils import remove_pt_head
    import torch

    model = torch.load(macefield_model_path, map_location="cpu", weights_only=False)
    if hasattr(model, "heads") and len(model.heads) != 1:
        model = remove_pt_head(model, "mp-dielectric")
    model = model.to(device="cpu", dtype=torch.float64)
    num_channels = model.node_embedding.linear.irreps_out.count("0e")
    model.products[0].linear.irreps_out = Irreps(
        f"{num_channels}x0e+{num_channels}x1o+{num_channels}x2e"
    )
    monkeypatch.setattr(model, "to", lambda *args, **kwargs: model)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: model)

    with pytest.raises(RuntimeError, match=r"scalar and vector features \(L_max=1\)"):
        extract_mace_data(
            macefield_model_path,
            species=[7, 13],
            head="mp-dielectric",
            num_spline_points=8,
        )


@pytest.mark.skipif(
    extract_mace_data is None,
    reason=f"extract_mace_data is not available: {extract_mace_data_import_error}",
)
def test_macefield_extractor_retains_legacy_pair_splines(macefield_model_path):
    data = extract_mace_data(
        macefield_model_path,
        species=[7, 13],
        head="mp-dielectric",
        num_spline_points=8,
        radial_format="pair-splines",
    )

    assert "symmetrix_format_version" not in data
    assert "compact_radial" not in data
    assert data["radial_spline_min"] == pytest.approx(1e-12)
    assert data["A0_spline_min"] == pytest.approx(1e-12)
    assert data["A1_spline_min"] == pytest.approx(1e-12)
    assert len(data["radial_spline_values_0"]) == 3
    assert len(data["radial_spline_values_1"]) == 3
    assert len(data["A0_spline_values"]) == 3
    assert len(data["A1_spline_values"]) == 3


@pytest.mark.skipif(
    extract_mace_data is None,
    reason=f"extract_mace_data is not available: {extract_mace_data_import_error}",
)
def test_legacy_pair_splines_retain_low_node_count_support(macefield_model_path):
    for num_spline_points in (2, 3):
        data = extract_mace_data(
            macefield_model_path,
            species=[7],
            head="mp-dielectric",
            num_spline_points=num_spline_points,
            radial_format="pair-splines",
        )

        assert len(data["radial_spline_values_0"][0][0]) == num_spline_points
        assert len(data["radial_spline_derivs_0"][0][0]) == num_spline_points
        assert len(data["A0_spline_values"][0]) == num_spline_points
        assert len(data["A1_spline_values"][0]) == num_spline_points


@pytest.mark.skipif(
    extract_mace_data is None,
    reason=f"extract_mace_data is not available: {extract_mace_data_import_error}",
)
def test_macefield_extractor_defaults_to_all_checkpoint_elements(macefield_model_path):
    data = extract_mace_data(
        macefield_model_path,
        head="mp-dielectric",
        num_spline_points=8,
    )

    assert len(data["atomic_numbers"]) == 79
    assert data["num_elements"] == 79
    assert data["atomic_numbers"] == sorted(data["atomic_numbers"])
    assert len(data["compact_radial"]["distance_transform"]["covalent_radii"]) == 79
    assert "radial_spline_values_0" not in data
    assert "radial_spline_values_1" not in data


@pytest.mark.skipif(
    extract_mace_data is None,
    reason=f"extract_mace_data is not available: {extract_mace_data_import_error}",
)
def test_compact_radial_values_and_derivatives_match_pair_splines(
    macefield_model_path,
    tmp_path,
):
    from symmetrix import symmetrix as native_symmetrix

    compact = extract_mace_data(
        macefield_model_path,
        species=[7, 13],
        head="mp-dielectric",
        num_spline_points=32,
    )
    legacy = extract_mace_data(
        macefield_model_path,
        species=[7, 13],
        head="mp-dielectric",
        num_spline_points=32,
        radial_format="pair-splines",
    )
    compact_path = tmp_path / "compact.json"
    legacy_path = tmp_path / "legacy.json"
    compact_path.write_text(json.dumps(compact, separators=(",", ":")))
    legacy_path.write_text(json.dumps(legacy))

    compact_model = native_symmetrix.MACE(str(compact_path))
    legacy_model = native_symmetrix.MACE(str(legacy_path))
    compact_model.prepare_active_types(
        np.asarray([0, 1, 0, 1], dtype=np.int32)[::2],
    )
    assert compact_model.active_atomic_numbers == [7]
    grid_min = compact["compact_radial"]["spline_grid_min"]
    h = legacy["radial_spline_h"]
    rng = np.random.default_rng(17)
    radii = np.concatenate(
        (
            grid_min + h * np.arange(32),
            rng.uniform(grid_min, compact_model.r_cut - h, size=17),
        )
    )
    node_types = np.asarray([0, 1], dtype=np.int32)
    num_neigh = np.asarray([len(radii), len(radii)], dtype=np.int32)
    neigh_types = np.concatenate(
        (
            np.full(len(radii), 1, dtype=np.int32),
            np.full(len(radii), 0, dtype=np.int32),
        )
    )
    pair_radii = np.tile(radii, 2)

    for method_name, values_name, derivatives_name in (
        ("compute_R0", "R0", "R0_deriv"),
        ("compute_R1", "R1", "R1_deriv"),
    ):
        for model in (compact_model, legacy_model):
            getattr(model, method_name)(
                2,
                node_types,
                num_neigh,
                neigh_types,
                pair_radii,
            )
        assert np.allclose(
            getattr(compact_model, values_name),
            getattr(legacy_model, values_name),
            rtol=0.0,
            atol=1e-11,
        )
        assert np.allclose(
            getattr(compact_model, derivatives_name),
            getattr(legacy_model, derivatives_name),
            rtol=0.0,
            atol=1e-9,
        )
        assert np.all(np.isfinite(getattr(compact_model, values_name)))
        assert np.all(np.isfinite(getattr(compact_model, derivatives_name)))
