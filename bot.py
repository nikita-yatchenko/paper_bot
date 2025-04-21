import asyncio
import os

from aiogram import Bot, Dispatcher

from src.services.file_processor.client import PaperProcessor
from src.settings.logger import setup_logger


async def start_bot():
    bot = Bot(token=os.getenv("BOT_TOKEN"))
    dp = Dispatcher()

    # Initialize services
    # db = Database(os.getenv("DATABASE_URL"))
    paper_processor = PaperProcessor()
    logger = setup_logger()

    # Inject dependencies
    dp["bot"] = bot
    # dp["db"] = db
    dp["paper_processor"] = paper_processor
    dp["logger"] = logger

    # Include routers
    from src.bot.routers import analyze, common
    dp.include_router(analyze.router)
    dp.include_router(common.router)

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(start_bot())
