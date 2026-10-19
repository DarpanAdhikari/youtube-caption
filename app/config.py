from pydantic import BaseModel


class Settings(BaseModel):
    default_languages: list[str] = ["ne", "en"]
    app_title: str = "YouTube Transcript API"
    app_version: str = "1.0.0"


settings = Settings()
