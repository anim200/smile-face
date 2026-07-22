"""Application configuration.

All settings are read from the environment (or a local ``.env``) exactly once
and cached, so no module needs to reach for ``os.environ`` directly.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    student_id: str = "30225"
    log_level: str = "INFO"

    postgres_user: str = "smile"
    postgres_password: str = "smile"
    postgres_db: str = "smiledb"
    postgres_host: str = "db"
    postgres_port: int = 5432

    base_dir: Path = Path("/app")
    data_dir: Path = Path("/app/data")
    model_dir: Path = Path("/app/models")
    seed_dir: Path = Path("/app/seed")

    #: Rejected before the file is read, so a large upload cannot exhaust memory.
    max_upload_bytes: int = 8 * 1024 * 1024
    max_files_per_upload: int = 200
    #: Below this many staged images per class, training stays disabled.
    min_images_per_class: int = 5
    #: Images are square; HOG cell geometry assumes a multiple of 8.
    image_size: int = 64
    #: A retrained model scoring below this is rejected rather than promoted.
    min_promote_accuracy: float = 0.55

    history_page_size: int = Field(default=20, ge=1, le=100)

    @computed_field
    @property
    def app_name(self) -> str:
        return f"Smile Classifier - {self.student_id}"

    @computed_field
    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def dataset_dir(self) -> Path:
        return self.data_dir / "dataset"

    @property
    def staging_dir(self) -> Path:
        return self.data_dir / "staging"

    @property
    def predictions_dir(self) -> Path:
        return self.data_dir / "predictions"

    def ensure_directories(self) -> None:
        """Create every runtime directory. Safe to call repeatedly."""
        for path in (
            self.data_dir,
            self.model_dir,
            self.dataset_dir,
            self.staging_dir,
            self.predictions_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings instance."""
    return Settings()