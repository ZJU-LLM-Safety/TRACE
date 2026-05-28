import os
from pathlib import Path

from openai import OpenAI


def load_local_env():
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def get_config_value(value, env_name, name, required=False, default=None):
    load_local_env()
    resolved = value if value not in (None, "") else os.getenv(env_name, default)
    if required and not resolved:
        raise ValueError(f"{name} is required. Pass it explicitly, set {env_name}, or add it to .env.")
    return resolved


def get_openai_client(openai_api_key=None, openai_base_url=None):
    api_key = get_config_value(
        openai_api_key,
        "OPENAI_API_KEY",
        "openai_api_key",
        required=True,
    )
    base_url = get_config_value(
        openai_base_url,
        "OPENAI_BASE_URL",
        "openai_base_url",
        default="https://api.openai.com/v1",
    )
    return OpenAI(api_key=api_key, base_url=base_url)


def ensure_parent_dir(path):
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
