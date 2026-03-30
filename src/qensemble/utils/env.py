import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env in repo root.
_DOTENV_PATH = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(dotenv_path=_DOTENV_PATH)


class RuntimeEnvironment(BaseModel):
    CUDA_VISIBLE_DEVICES: str = Field(default="")

    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
    }


_env_instance: RuntimeEnvironment | None = None


def get_env() -> RuntimeEnvironment:
    global _env_instance
    if _env_instance is not None:
        return _env_instance

    env_data = {
        key: value
        for key in ["CUDA_VISIBLE_DEVICES"]
        if (value := os.environ.get(key)) is not None
    }
    _env_instance = RuntimeEnvironment.model_validate(env_data)
    return _env_instance
