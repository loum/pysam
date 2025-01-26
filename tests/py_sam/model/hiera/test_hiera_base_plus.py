"""Model definition for Segment Anything Model 2 (SAM 2) Hiera Base+ checkpoint unit tests."""

from py_sam.model.hiera import HieraBasePlus


def test_hiera_base_plus_init() -> None:
    """Initialise a HieraBasePlus object."""
    # Given an initialised a HieraBasePlus
    model = HieraBasePlus()

    # I should get a HieraBasePlus instance
    assert isinstance(model, HieraBasePlus), "Object is not a HieraBasePlus instance"

    # and the attributes should match
    assert model.model_type == "hiera_b", "HieraB instance model type error"
    assert (
        model.repo_id == "facebook/sam2-hiera-base-plus"
    ), "HieraB instance repo_id error"
    assert model.filename == "sam2_hiera_base_plus.pt", "HieraB instance filename error"
    assert model.model_cfg == "sam2_hiera_b+.yaml", "HieraB instance filename error"
