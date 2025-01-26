"""Model definition for Segment Anything Model 2 (SAM 2) Hiera Tiny checkpoint unit tests."""

from py_sam.model.hiera import HieraTiny


def test_hiera_tiny_init() -> None:
    """Initialise a HieraTiny object."""
    # Given an initialised a HieraTiny
    model = HieraTiny()

    # I should get a HiearaTiny instance
    assert isinstance(model, HieraTiny), "Object is not a HieraTiny instance"

    # and the attributes should match
    assert model.model_type == "hiera_t", "HieraT instance model type error"
    assert model.repo_id == "facebook/sam2-hiera-tiny", "HieraT instance repo_id error"
    assert model.filename == "sam2_hiera_tiny.pt", "HieraT instance filename error"
    assert model.model_cfg == "sam2_hiera_t.yaml", "HieraT instance filename error"
