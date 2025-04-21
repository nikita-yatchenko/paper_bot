import base64
import io
import re

from PIL import Image


def looks_like_base64(sb):
    """
    Проверяет, выглядит ли строка как base64.

    Аргументы:
    sb: Строка для проверки.

    Возвращает:
    True, если строка выглядит как base64, иначе False.
    """
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None


def is_image_data(b64data):
    """
    Проверяет, является ли base64 данные изображением, проверяя сигнатуры данных.

    Аргументы:
    b64data: Строка base64, представляющая изображение.

    Возвращает:
    True, если данные начинаются с сигнатуры изображения, иначе False.
    """
    image_signatures = {
        b"\xFF\xD8\xFF": "jpg",
        b"\x89\x50\x4E\x47\x0D\x0A\x1A\x0A": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]
        for sig, format in image_signatures.items():
            if header.startswith(sig):
                return True
        return False
    except Exception:
        return False


def resize_base64_image(base64_string, size=(128, 128)):
    """
    Изменяет размер изображения, закодированного в формате base64.

    Аргументы:
    base64_string: Строка base64, представляющая изображение.
    size: Новый размер изображения.

    Возвращает:
    Закодированное в формате base64 изображение нового размера.
    """
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))

    # Изменение размера изображения с использованием алгоритма LANCZOS для улучшения качества
    resized_img = img.resize(size, Image.LANCZOS)

    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)

    return base64.b64encode(buffered.getvalue()).decode("utf-8")
