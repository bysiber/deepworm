"""Configuration for DeepWorm."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# Config file names to search for (in order of priority)
CONFIG_FILES = [
    "deepworm.toml",
    ".deepworm.toml",
    "deepworm.yaml",
    "deepworm.yml",
    ".deepworm.yaml",
    ".deepworm.yml",
    "pyproject.toml",  # [tool.deepworm] section
]


@dataclass
class Config:
    """Research configuration."""

    # LLM settings
    provider: str = "openai"
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.1

    # Research settings
    depth: int = 1
    breadth: int = 1
    max_sources: int = 10

    # Output settings
    output_format: str = "markdown"
    output_file: Optional[str] = None
    verbose: bool = False

    # Search settings
    search_region: str = "wt-wt"
    search_max_results: int = 8
    search_provider: str = "duckduckgo"  # duckduckgo, brave, searxng, tavily

    # Rate limiting
    max_requests_per_minute: int = 60  # LLM API rate limit
    timeout_seconds: int = 0  # Overall research time budget (0 = unlimited)

    def __post_init__(self):
        if self.api_key is None:
            self._detect_provider()

        if self.model is None:
            self.model = self._default_model()

        self.validate()

    @classmethod
    def from_env(cls, **overrides: Any) -> "Config":
        """Create config with environment variable overrides.

        Environment variables follow the pattern DEEPWORM_<FIELD>, e.g.:
        - DEEPWORM_PROVIDER=anthropic
        - DEEPWORM_DEPTH=3
        - DEEPWORM_BREADTH=6
        - DEEPWORM_MODEL=gpt-4o
        - DEEPWORM_OUTPUT_FORMAT=html
        - DEEPWORM_VERBOSE=true

        Explicit overrides take precedence over env vars.
        """
        env_config = _load_env_overrides()
        env_config.update(overrides)
        return cls(**env_config)

    def _detect_provider(self):
        """Auto-detect provider from saved keys file or environment variables."""
        # Load saved keys from ~/.deepworm_keys first
        _load_saved_keys()

        if os.getenv("OPENROUTER_API_KEY"):
            self.provider = "openrouter"
            self.api_key = os.getenv("OPENROUTER_API_KEY")
            self.base_url = "https://openrouter.ai/api/v1"
        elif os.getenv("OPENAI_API_KEY"):
            self.provider = "openai"
            self.api_key = os.getenv("OPENAI_API_KEY")
        elif os.getenv("ANTHROPIC_API_KEY"):
            self.provider = "anthropic"
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
        elif os.getenv("GOOGLE_API_KEY"):
            self.provider = "google"
            self.api_key = os.getenv("GOOGLE_API_KEY")
        else:
            # Default to Ollama (local, no API key needed)
            self.provider = "ollama"
            self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            self.api_key = "ollama"

    def _default_model(self, provider: str | None = None) -> str:
        """Return default model for the provider."""
        defaults = {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-5-haiku-latest",
            "google": "gemini-2.5-flash-lite",
            "openrouter": "google/gemini-2.0-flash-001",
            "ollama": "llama3.2",
        }
        return defaults.get(provider or self.provider, "gpt-4o-mini")

    @property
    def ollama_base_url(self) -> str:
        return self.base_url or "http://localhost:11434/v1"

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If any configuration value is invalid.
        """
        VALID_PROVIDERS = {"openai", "anthropic", "google", "openrouter", "ollama"}
        VALID_FORMATS = {"markdown", "html", "text", "json", "pdf"}
        VALID_SEARCH_PROVIDERS = {"duckduckgo", "brave", "searxng", "tavily"}

        if self.provider not in VALID_PROVIDERS:
            raise ValueError(
                f"Invalid provider '{self.provider}'. "
                f"Must be one of: {', '.join(sorted(VALID_PROVIDERS))}"
            )
        if self.depth < 1 or self.depth > 20:
            raise ValueError(f"depth must be between 1 and 20, got {self.depth}")
        if self.breadth < 1 or self.breadth > 20:
            raise ValueError(f"breadth must be between 1 and 20, got {self.breadth}")
        if self.max_sources < 1 or self.max_sources > 100:
            raise ValueError(f"max_sources must be between 1 and 100, got {self.max_sources}")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be between 0.0 and 2.0, got {self.temperature}")
        if self.output_format not in VALID_FORMATS:
            raise ValueError(
                f"Invalid output_format '{self.output_format}'. "
                f"Must be one of: {', '.join(sorted(VALID_FORMATS))}"
            )
        if self.search_provider not in VALID_SEARCH_PROVIDERS:
            raise ValueError(
                f"Invalid search_provider '{self.search_provider}'. "
                f"Must be one of: {', '.join(sorted(VALID_SEARCH_PROVIDERS))}"
            )
        if self.search_max_results < 1 or self.search_max_results > 50:
            raise ValueError(
                f"search_max_results must be between 1 and 50, got {self.search_max_results}"
            )
        if self.max_requests_per_minute < 1:
            raise ValueError(
                f"max_requests_per_minute must be >= 1, got {self.max_requests_per_minute}"
            )
        if self.timeout_seconds < 0:
            raise ValueError(
                f"timeout_seconds must be >= 0, got {self.timeout_seconds}"
            )

    @classmethod
    def auto(cls) -> "Config":
        """Create a Config with auto-detected settings.

        Priority: config file < environment variables < auto-detect.
        """
        file_config = _load_config_file() or {}
        env_config = _load_env_overrides()
        merged = {**file_config, **env_config}
        if merged:
            return cls(**merged)
        return cls()

    @classmethod
    def from_file(cls, path: str) -> "Config":
        """Load config from a specific TOML or YAML file."""
        p = Path(path)
        if p.suffix in (".yaml", ".yml"):
            data = _parse_yaml_file(p)
        else:
            data = _parse_toml_file(p)
        if data is None:
            raise FileNotFoundError(f"Config file not found: {path}")
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load config from a YAML file."""
        data = _parse_yaml_file(Path(path))
        if data is None:
            raise FileNotFoundError(f"YAML config file not found: {path}")
        return cls(**data)


def _load_config_file() -> dict[str, Any] | None:
    """Search for and load a config file from CWD up to root."""
    cwd = Path.cwd()

    for directory in [cwd, *cwd.parents]:
        for filename in CONFIG_FILES:
            filepath = directory / filename
            if filepath.exists():
                if filepath.suffix in (".yaml", ".yml"):
                    return _parse_yaml_file(filepath)
                return _parse_toml_file(filepath)
    return None


def _parse_toml_file(path: Path) -> dict[str, Any] | None:
    """Parse a TOML config file and extract deepworm settings."""
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            # Python < 3.11 without tomli installed
            return None

    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except Exception:
        return None

    # Check for [tool.deepworm] in pyproject.toml
    if path.name == "pyproject.toml":
        data = data.get("tool", {}).get("deepworm", {})
        if not data:
            return None

    # Filter to only valid Config fields
    valid_fields = {f.name for f in Config.__dataclass_fields__.values()}
    return {k: v for k, v in data.items() if k in valid_fields} or None


# Environment variable prefix for config overrides
_ENV_PREFIX = "DEEPWORM_"

# Mapping from env var suffix to (field_name, type_converter)
_ENV_FIELD_MAP: dict[str, tuple[str, type]] = {
    "PROVIDER": ("provider", str),
    "MODEL": ("model", str),
    "API_KEY": ("api_key", str),
    "BASE_URL": ("base_url", str),
    "TEMPERATURE": ("temperature", float),
    "DEPTH": ("depth", int),
    "BREADTH": ("breadth", int),
    "MAX_SOURCES": ("max_sources", int),
    "OUTPUT_FORMAT": ("output_format", str),
    "OUTPUT_FILE": ("output_file", str),
    "VERBOSE": ("verbose", bool),
    "SEARCH_REGION": ("search_region", str),
    "SEARCH_MAX_RESULTS": ("search_max_results", int),
    "SEARCH_PROVIDER": ("search_provider", str),
    "MAX_REQUESTS_PER_MINUTE": ("max_requests_per_minute", int),
    "TIMEOUT_SECONDS": ("timeout_seconds", int),
}


def _load_env_overrides() -> dict[str, Any]:
    """Load config overrides from DEEPWORM_* environment variables.

    Returns:
        Dictionary of field_name → converted_value for any set env vars.
    """
    overrides: dict[str, Any] = {}

    for suffix, (field_name, converter) in _ENV_FIELD_MAP.items():
        value = os.getenv(f"{_ENV_PREFIX}{suffix}")
        if value is None:
            continue

        try:
            if converter is bool:
                overrides[field_name] = value.lower() in ("1", "true", "yes", "on")
            elif converter is int:
                overrides[field_name] = int(value)
            elif converter is float:
                overrides[field_name] = float(value)
            else:
                overrides[field_name] = value
        except (ValueError, TypeError):
            # Skip malformed env var values
            continue

    return overrides
def _parse_yaml_file(path: Path) -> dict[str, Any] | None:
    """Parse a YAML config file and extract deepworm settings."""
    try:
        import yaml
    except ImportError:
        return None

    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    # Support nested deepworm key or flat config
    if "deepworm" in data and isinstance(data["deepworm"], dict):
        data = data["deepworm"]

    # Filter to only valid Config fields
    valid_fields = {f.name for f in Config.__dataclass_fields__.values()}
    return {k: v for k, v in data.items() if k in valid_fields} or None


# ---------------------------------------------------------------------------
# Saved API keys management (~/.deepworm_keys)
# ---------------------------------------------------------------------------

_KEYS_FILE = os.path.expanduser("~/.deepworm_keys")

# Mapping: provider name → environment variable name
PROVIDER_KEY_ENVS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def _load_saved_keys() -> None:
    """Load saved API keys from ~/.deepworm_keys into environment variables.

    Keys are only loaded if the corresponding env var is not already set.
    File format: one ``ENV_VAR=value`` per line.
    """
    if not os.path.exists(_KEYS_FILE):
        return
    try:
        with open(_KEYS_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                # Only set if not already in environment
                if value and not os.getenv(key):
                    os.environ[key] = value
    except OSError:
        pass


def save_api_key(provider: str, api_key: str) -> None:
    """Save an API key for *provider* to ~/.deepworm_keys.

    If the provider already has a key in the file it is updated in-place.
    The key is also injected into the current process environment.
    """
    env_var = PROVIDER_KEY_ENVS.get(provider)
    if not env_var:
        return

    # Update current process env
    os.environ[env_var] = api_key

    # Read existing file
    lines: list[str] = []
    found = False
    if os.path.exists(_KEYS_FILE):
        with open(_KEYS_FILE, "r") as f:
            for line in f:
                if line.strip().startswith(f"{env_var}="):
                    lines.append(f"{env_var}={api_key}\n")
                    found = True
                else:
                    lines.append(line)
    if not found:
        lines.append(f"{env_var}={api_key}\n")

    with open(_KEYS_FILE, "w") as f:
        f.writelines(lines)

    # Restrict permissions (owner-only read/write)
    os.chmod(_KEYS_FILE, 0o600)


def get_saved_keys_status() -> dict[str, bool]:
    """Return which providers have API keys configured (env or file)."""
    _load_saved_keys()
    return {
        provider: bool(os.getenv(env_var))
        for provider, env_var in PROVIDER_KEY_ENVS.items()
    }
