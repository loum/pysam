"""Model definition for Segment Anything Model 2 (SAM 2) Hiera Tiny checkpoint."""

from py_sam.model import Model


class HieraTiny(Model):
    """SAM 2 Hiera Tiny (Hiera-T) checkpoint."""

    def __init__(self) -> None:
        """Initialise a HieraT instance."""
        self.__model_type = "hiera_t"
        self.__repo_id = "facebook/sam2-hiera-tiny"
        self.__filename = "sam2_hiera_tiny.pt"
        self.__model_cfg = "sam2_hiera_t.yaml"

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
