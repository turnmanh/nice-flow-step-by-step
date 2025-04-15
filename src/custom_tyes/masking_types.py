from pydantic import BaseModel, field_validator


class MaskingType(BaseModel):
    mask: str = "checkerboard"

    @field_validator("mask")
    def validate_masking_type(value) -> str:
        """Validates the masking type.

        Args:
            value: Masking type.

        Returns:
            Masking type.
        """
        if value not in ["checkerboard", "stripes"]:
            raise ValueError(f"Unknown masking type: {value}")
        return value
