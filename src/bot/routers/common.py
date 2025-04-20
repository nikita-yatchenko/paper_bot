import logging
from asyncio import Lock

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from src.services.handlers.commands import contact_us
from src.services.handlers.keyboards import main_menu

router = Router()
state_lock = Lock()


@router.message(Command("start"))
async def start(message: Message, logger: logging.Logger):
    user_id = message.from_user.id
    logger.info(f"User {user_id} started the bot", extra={'user_id': user_id})
    await message.answer(
        "Welcome to AI Avatar Bot! 🎨\n"
        "Use /train to create your model\n"
        "Use /generate to create images",
        reply_markup=main_menu()
    )


@router.message(Command("contact_us"))
async def cmd_contact_us(message: Message) -> None:
    """Contact us handler."""
    await contact_us(message)
