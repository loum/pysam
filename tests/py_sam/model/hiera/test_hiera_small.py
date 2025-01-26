"""Model definition for Segment Anything Model 2 (SAM 2) Hiera Small checkpoint unit tests."""

from py_sam.model.hiera import HieraSmall


def test_hiera_small_init() -> None:
    """Initialise a HieraSmall object."""
    # Given an initialised a HieraSmall
    model = HieraSmall()

    # I should get a HieraSmall instance
    assert isinstance(model, HieraSmall), "Object is not a HieraSmall instance"

    # and the attributes should match
    assert model.model_type == "hiera_s", "HieraS instance model type error"
    assert model.repo_id == "facebook/sam2-hiera-small", "HieraS instance repo_id error"
    assert model.filename == "sam2_hiera_small.pt", "HieraS instance filename error"
    assert model.model_cfg == "sam2_hiera_s.yaml", "HieraS instance filename error"
