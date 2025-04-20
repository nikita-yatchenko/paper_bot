from pydantic_settings import BaseSettings


class LinkSettings(BaseSettings):
    """Links."""
    channel: str = "None"
    feedback_chat_id: int | None = None
    contact_us: str = "https://t.me/selfieai_ask"
    # site: str = "None"

    # model_config = SettingsConfigDict(env_prefix="link_")
