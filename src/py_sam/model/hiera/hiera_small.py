"""Model definition for Segment Anything Model 2 (SAM 2) Hiera Small checkpoint."""

from py_sam.model import Model


class HieraSmall(Model):
    """SAM 2 Hiera Small (Hiera-S) checkpoint."""

    def __init__(self) -> None:
        """Initialise a HieraS instance."""
        self.__model_type = "hiera_s"
        self.__repo_id = "facebook/sam2-hiera-small"
        self.__filename = "sam2_hiera_small.pt"
        self.__model_cfg = "sam2_hiera_s.yaml"

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
