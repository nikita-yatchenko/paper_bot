from aiogram.fsm.state import State, StatesGroup


class AnalyzingStates(StatesGroup):
    AWAITING_PAPER = State()
    AWAITING_ANALYSIS = State()
    CONFIRMATION = State()
