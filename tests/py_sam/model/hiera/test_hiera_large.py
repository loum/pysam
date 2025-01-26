"""Model definition for Segment Anything Model 2 (SAM 2) Hiera Large checkpoint unit tests."""

from py_sam.model.hiera import HieraLarge


def test_hiera_large_init() -> None:
    """Initialise a HieraLarge object."""
    # Given an initialised a HieraLarge
    model = HieraLarge()

    # I should get a HieraLarge instance
    assert isinstance(model, HieraLarge), "Object is not a HieraLarge instance"

    # and the attributes should match
    assert model.model_type == "hiera_l", "HieraL instance model type error"
    assert model.repo_id == "facebook/sam2-hiera-large", "HieraL instance repo_id error"
    assert model.filename == "sam2_hiera_large.pt", "HieraL instance filename error"
    assert model.model_cfg == "sam2_hiera_l.yaml", "HieraL instance filename error"
