from aiogram.enums import ParseMode
from aiogram.types import Message

from src.settings.links import LinkSettings

links = LinkSettings()


async def contact_us(message: Message) -> None:
    """Contact us."""
    await message.answer(
        f"[Связаться с нами]({links.contact_us})",
        parse_mode=ParseMode.MARKDOWN_V2,
    )
