from aiogram.utils.keyboard import ReplyKeyboardBuilder


def main_menu():
    builder = ReplyKeyboardBuilder()
    builder.button(text="/analyze")
    # builder.button(text="/generate summary")
    # builder.button(text="/balance")
    builder.button(text="/contact_us")
    builder.adjust(2)
    return builder.as_markup(resize_keyboard=True)


def yes_no_keyboard():
    """Build a Yes/No keyboard."""
    builder = ReplyKeyboardBuilder()
    builder.button(text="Yes")
    builder.button(text="No")
    builder.adjust(2)  # Arrange buttons in 2 columns
    return builder.as_markup(resize_keyboard=True)  # Make the keyboard compact
