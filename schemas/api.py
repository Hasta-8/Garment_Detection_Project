"""API contracts: what /v1/analyze returns, and the one error shape."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Alternative(BaseModel):
    label: str
    score: float = Field(ge=0, le=1)


class AttributeResult(BaseModel):
    label: str = Field(description="The highest-scoring candidate")
    score: float = Field(
        ge=0,
        le=1,
        description="Softmax of Fashion-CLIP similarities over this attribute's candidates: "
        "a relative score, not a calibrated probability",
    )
    alternatives: list[Alternative] = Field(description="Top candidates, best first")


class Attributes(BaseModel):
    category: AttributeResult
    colour: AttributeResult
    pattern: AttributeResult
    sleeve: AttributeResult
    gender: AttributeResult = Field(
        description="Catalogue section the garment is styled for (men / women / unisex), "
        "inferred from the image; not a statement about any person"
    )


class ImageInfo(BaseModel):
    format: str
    width: int
    height: int
    exif_orientation: int


class AnalyzeResponse(BaseModel):
    request_id: str
    attributes: Attributes
    title: str = Field(description="Assembled from the labels by a fixed template")
    description: str = Field(description="Assembled from the labels by a fixed template")
    image: ImageInfo
    timings: dict[str, float]
    model: str
    settings: dict[str, int | float]


class ErrorResponse(BaseModel):
    error: str = Field(description="Stable machine-readable code, e.g. unsupported_type")
    detail: str
    request_id: str
