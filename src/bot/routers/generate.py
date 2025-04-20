import logging

from aiogram import Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from src.services.handlers.generate import start_generating
from src.services.handlers.keyboards import main_menu
from src.services.vlm.client import MLClient

router = Router()


@router.message(Command("analyze_paper"))
async def analyze_paper(message: Message, state: FSMContext, ml_client: MLClient, logger: logging.Logger):
    user_id = message.from_user.id
    logger.info(f"User {user_id} initiated generation", extra={'user_id': user_id})

    training_job_id = ml_client.redis_cache.get(message.from_user.id)
    if not training_job_id:
        await message.answer("Please train a model first using /train", reply_markup=main_menu())
        return

    status = ml_client._get_job_status(training_job_id)
    if status.status != "completed":
        await message.answer("Please wait - your model is being trained", reply_markup=main_menu())
        return

    current_state = await state.get_state()
    if current_state:
        logger.info(f"Clearing state for user {message.from_user.id} (current state: {current_state})",
                    extra={'user_id': message.from_user.id})
        await state.clear()

    # await state.set_state(GeneratingStates.GENERATING_IMAGES)
    # logger.debug(f"State set to: {GeneratingStates.GENERATING_IMAGES}")
    # await message.answer("Please enter your prompt for image generation:")

    await start_generating(
        ml_client=ml_client,
        message=message
    )


# @router.message(GeneratingStates.GENERATING_IMAGES)
# async def handle_generation(message: Message, ml_client: MLClient, logger: logging.Logger):
#     user_id = message.from_user.id
#     logger.debug(f"User {user_id} generation")
#
#     await start_generating(
#         ml_client=ml_client,
#         message=message
#     )
