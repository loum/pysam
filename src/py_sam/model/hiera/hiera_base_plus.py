"""Model definition for Segment Anything Model 2 (SAM 2) Hiera Base+ checkpoint."""

from py_sam.model import Model


class HieraBasePlus(Model):
    """SAM 2 Hiera Base+ (Hiera-B) checkpoint."""

    def __init__(self) -> None:
        """Initialise a HieraB instance."""
        self.__model_type = "hiera_b"
        self.__repo_id = "facebook/sam2-hiera-base-plus"
        self.__filename = "sam2_hiera_base_plus.pt"
        self.__model_cfg = "sam2_hiera_b+.yaml"

        super().__init__()

    @property
    def model_type(self) -> str:
        """Getter for the `model_type` attribute."""
        return self.__model_type

    @property
    def repo_id(self) -> str:
        """Getter for the `repo_id` attribute."""
        return self.__repo_id

    @property
    def filename(self) -> str:
        """Getter for the `filename` attribute."""
        return self.__filename

    @property
    def model_cfg(self) -> str:
        """Getter for the `model_cfg` attribute."""
        return self.__model_cfg
