import logging
import os
from asyncio import Lock

from aiogram import Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from src.services.handlers.keyboards import main_menu, yes_no_keyboard
from src.services.handlers.states import TrainingStates
from src.services.handlers.train import start_training
from src.services.ml.client import MLClient

router = Router()
state_lock = Lock()


@router.message(Command("train"))
async def train(message: Message, state: FSMContext, logger: logging.Logger):
    user_id = message.from_user.id
    logger.info(f"User {user_id} initiated training", extra={'user_id': user_id})

    # Clear any existing photos and media groups from state
    await state.update_data(photos=set(), media_groups={}, message_ids=set())

    await message.answer(
        f"📸 Please upload exactly {os.environ["MIN_PHOTOS"]} clear face photos\n"
        "Tips:\n- Different angles\n- Good lighting\n- No sunglasses",
        reply_markup=main_menu()
    )
    await state.set_state(TrainingStates.AWAITING_IMAGES)
    await state.update_data(media_groups={})  # Initialize media group tracking


@router.message(TrainingStates.AWAITING_IMAGES)
async def handle_photos(message: Message, state: FSMContext, logger: logging.Logger):
    logger.debug(f"User {message.from_user.id} handling photos")

    async with state_lock:  # Ensure atomic state updates
        data = await state.get_data()
        photos = data.get("photos", set())  # List of all uploaded photos

        # Handle media group
        if message.media_group_id:
            media_group_id = message.media_group_id
            media_groups = data.get("media_groups", {})

            # Initialize group if it doesn't exist
            if media_group_id not in media_groups:
                media_groups[media_group_id] = set()

            # Add current file to the group
            if message.photo:
                media_groups[media_group_id].add(message.photo[-1].file_id)  # Highest resolution
            elif message.document and message.document.mime_type.startswith('image/'):
                media_groups[media_group_id].add(message.document.file_id)

            # Update media groups in state
            await state.update_data(media_groups=media_groups)

            photos.update(media_groups[media_group_id])

        # Handle individual upload
        else:
            if message.photo:
                photos.add(message.photo[-1].file_id)  # Highest resolution
            elif message.document and message.document.mime_type.startswith('image/'):
                photos.add(message.document.file_id)
            else:
                await message.answer("Please upload valid images (photos or image files).")
                return

        # Update photos in state
        await state.update_data(photos=photos)

        logger.debug(f"User {message.from_user.id} processed: {photos}")
        if len(photos) >= int(os.environ["MIN_PHOTOS"]):
            await state.set_state(TrainingStates.CONFIRMATION)
            await message.answer(
                f"Generate with {len(photos)} photos provided? (Yes/No)",
                reply_markup=yes_no_keyboard()
            )


@router.message(TrainingStates.CONFIRMATION)
async def handle_confirmation(message: Message, state: FSMContext, ml_client: MLClient, logger: logging.Logger) -> None:
    logger.debug(f"User {message.from_user.id} confirmation")
    user_id = message.from_user.id
    response = message.text.lower()

    if response == "yes":
        # Start training with the current photos
        data = await state.get_data()
        photos = set(data.get("photos", []))
        await start_training(
            ml_client=ml_client,
            user_id=user_id,
            file_ids=photos,
            state=state,
            message=message
        )
    elif response == "no":
        # Clear the state and start over
        await state.clear()
        await message.answer(
            "Starting over.",
            reply_markup=main_menu()  # Remove the Yes/No keyboard
        )
    else:
        await message.answer("Please respond with 'Yes' or 'No'.", reply_markup=yes_no_keyboard())
