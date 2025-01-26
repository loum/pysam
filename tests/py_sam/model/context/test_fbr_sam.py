"""Model definition for Segment Anything Model 2 (SAM 2) checkpoint context unit tests."""

from py_sam.model.context import FbrSam


def test_fbr_sam_init() -> None:
    """Initialise a Facebook Research context-based model lookup object."""
    assert (
        FbrSam.HIERA_B.value().model_type == "hiera_b"
    ), "FBR Hiera Base+ model context error for model_type."
    assert (
        FbrSam.HIERA_L.value().model_type == "hiera_l"
    ), "FBR Hiera Large model context error for model_type."
    assert (
        FbrSam.HIERA_S.value().model_type == "hiera_s"
    ), "FBR Hiera Small model context error for model_type."
    assert (
        FbrSam.HIERA_T.value().model_type == "hiera_t"
    ), "FBR Hiera Tiny model context error for model_type."
