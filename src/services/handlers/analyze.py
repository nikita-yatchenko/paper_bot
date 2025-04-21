import os

from aiogram.types import Message
from dotenv import find_dotenv, load_dotenv

from src.services.file_processor.client import PaperProcessor
from src.settings.logger import setup_logger

load_dotenv(find_dotenv())
mock = os.getenv("TEST") == "True"
logger = setup_logger()


async def process_user_response(
        paper_client: PaperProcessor,
        message: Message,
        paper_id: str
):
    """React to user response."""
    username = message.from_user.username

    assistant_answer = paper_client.respond(message.text, paper_id)
    logger.info(f"Responding to {username}: {assistant_answer}")
    await message.answer(assistant_answer)
