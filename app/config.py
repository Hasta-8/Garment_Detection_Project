"""Settings, loaded once from environment variables (prefix GARMENT_) and `.env`.

The model lives in a shared folder (GARMENT_MODELS_DIR, Hugging Face cache layout) so
several projects can reuse one copy; the revision is pinned so the weights can't change
underneath the code.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

MODEL = "patrickjohncyh/fashion-clip"  # MIT licence
MODEL_REVISION = "7e3ba62ce16b379a1ab479346b66f192e76f51b7"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="GARMENT_", extra="ignore")

    models_dir: Path = Path(".cache/models")
    demo_dir: Path = Path("demo")  # the demo page and its sample photos (optional)

    # How many ranked alternatives each attribute returns (the top label included).
    top_k: int = Field(default=3, ge=1, le=20)
    # Results can differ in the last digits between thread counts, so it is fixed.
    torch_threads: int = Field(default=4, ge=1)

    # Upload limits
    max_upload_mb: float = Field(default=10.0, gt=0)
    max_pixels: int = Field(default=40_000_000, gt=0)
    min_side_px: int = Field(default=32, ge=1)
    busy_timeout_s: float = Field(default=30.0, gt=0)


@lru_cache
def get_settings() -> Settings:
    return Settings()
