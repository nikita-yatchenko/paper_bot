from langchain.text_splitter import CharacterTextSplitter
from unstructured.partition.pdf import partition_pdf


# Функция извлечения элементов из pdf-файла
def extract_pdf_elements(path, fname):
    """
    Функция для извлечения различных элементов из PDF-файла, таких как изображения, таблицы,
    и текста. Также осуществляется разбиение текста на части (чанки) для дальнейшей обработки.

    Аргументы:
    path: Строка, содержащая путь к директории, в которую будут сохранены извлеченные изображения.
    fname: Строка, содержащая имя PDF-файла, который необходимо обработать.

    Возвращает:
    Список объектов типа `unstructured.documents.elements`, представляющих извлеченные из PDF элементы.
    """
    return partition_pdf(
        filename=path + fname,
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3000,
        combine_text_under_n_chars=2000,
        image_output_dir_path=path,
    )


# Функция категоризации элементов
def categorize_elements(raw_pdf_elements):
    """
    Функция для категоризации извлеченных элементов из PDF-файла.
    Элементы делятся на текстовые элементы и таблицы.

    Аргументы:
    raw_pdf_elements: Список объектов типа `unstructured.documents.elements`,
                      представляющих извлеченные из PDF элементы.

    Возвращает:
    Два списка: texts (текстовые элементы) и tables (таблицы).
    """
    tables = []  # Список для хранения элементов типа "таблица"
    texts = []   # Список для хранения текстовых элементов
    for element in raw_pdf_elements:
        # Проверка типа элемента. Если элемент является таблицей, добавляем его в список таблиц
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element))
        # Если элемент является композитным текстовым элементом, добавляем его в список текстов
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))
    return texts, tables  # Возвращаем списки с текстами и таблицами


# Указываем путь до директории, где находится PDF-файл
fpath = "/content/"
# Указываем имя PDF-файла, который нужно обработать
fname = "cj.pdf"

# Извлекаем элементы из PDF-файла с помощью функции extract_pdf_elements
raw_pdf_elements = extract_pdf_elements(fpath, fname)

# Категоризируем извлеченные элементы на текстовые и табличные с помощью функции categorize_elements
texts, tables = categorize_elements(raw_pdf_elements)

# Создаем объект CharacterTextSplitter для разбиения текста на части (чанки)
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=4000,
    chunk_overlap=1000
)

# Объединяем все текстовые элементы в одну строку
joined_texts = " ".join(texts)

# Разбиваем объединенный текст на чанки, используя созданный CharacterTextSplitter
texts_4k_token = text_splitter.split_text(joined_texts)
