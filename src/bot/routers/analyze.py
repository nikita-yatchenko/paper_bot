from aiogram import Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from src.services.file_processor.client import PaperProcessor
from src.services.handlers.analyze import process_user_response
from src.services.handlers.keyboards import main_menu
from src.services.handlers.states import AnalyzingStates
from src.settings.logger import setup_logger

router = Router()
logger = setup_logger()


@router.message(Command("analyze"))
async def analyze(message: Message, state: FSMContext):
    # Clear any existing paper_id
    await state.update_data(paper_id=None)

    await message.answer(
        "📸 Please provide an ArXiv paper id, like 2307.00651v1\n",
        reply_markup=main_menu()
    )
    await state.set_state(AnalyzingStates.AWAITING_PAPER)


@router.message(AnalyzingStates.AWAITING_PAPER)
async def process_paper(message: Message, state: FSMContext, paper_processor: PaperProcessor):
    user_id = message.from_user.id
    paper_id = message.text
    logger.info(f"User {user_id} initiated paper analysis of {paper_id}")
    await paper_processor.process(paper_id)
    logger.info(f"Done with paper analysis of {paper_id} for user {user_id}")
    await state.set_state(AnalyzingStates.CONFIRMATION)


@router.message(AnalyzingStates.CONFIRMATION)
async def handle_user_response(message: Message,
                               state: FSMContext,
                               paper_processor: PaperProcessor) -> None:
    logger.info(f"User {message.from_user.id} processing question: {message.text}")
    paper_id = state.get_data()["paper_id"]
    response = await process_user_response(paper_processor, message, paper_id)
    await message.answer(
            response,
            # reply_markup=main_menu()
        )
