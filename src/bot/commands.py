from aiogram import Bot
from aiogram.types import BotCommand, BotCommandScopeDefault


async def set_commands(bot: Bot):
    await bot.set_my_commands(
        [
            BotCommand(
                command='start',
                description='Запуск бота',
            ),
            BotCommand(
                command='train',
                description='Обучить свою модель',
            ),
            BotCommand(
                command='generate',
                description='Сгенерировать фотографию',
            ),
            BotCommand(
                command='contact_us',
                description='Связаться с нами',
            ),
        ],
        BotCommandScopeDefault(),
    )
