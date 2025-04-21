import os
import uuid
from pathlib import Path
from typing import List

import numpy as np
import requests
import torch
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from PIL import Image
from transformers import AutoModel, CLIPModel, CLIPProcessor, pipeline
from unstructured.partition.pdf import partition_pdf

from src.services.file_processor.utils import (
    is_image_data,
    looks_like_base64,
    resize_base64_image,
)
from src.services.vlm.client import CustomLLM
from src.settings.logger import setup_logger

logger = setup_logger()
project_root = Path(Path(os.path.abspath(__file__))).parent.parent.parent.parent
data_folder = project_root / "data"


class MultimodalEmbeddingFunction:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize the multimodal embedding function.

        Args:
            model_name (str): The name of the pretrained multimodal model.
        """
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.model_name = model_name
        logger.info("Loading model")
        if model_name == "openai/clip-vit-base-patch32":
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            logger.info("Loading processor")
            self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=False)
        else:
            self.model = AutoModel.from_pretrained("jinaai/jina-clip-v1",
                                                   trust_remote_code=True)

    def embed_documents(self, inputs: List) -> List[List[float]]:
        """
        Generate embeddings for a list of text and/or image inputs.

        Args:
            inputs (list): A list of inputs, where each input is either:
                - A string (text)
                - A PIL.Image object (image)

        Returns:
            List[List[float]]: A list of embeddings as lists of floats.
        """
        # Generate embeddings using the multimodal function
        embeddings_tensor = self._multimodal_embedding_function(inputs)
        return embeddings_tensor.tolist()

    def embed_query(self, input_item) -> List[float]:
        """
        Generate an embedding for a single text or image input.

        Args:
            input_item: Either a string (text) or a PIL.Image object (image).

        Returns:
            List[float]: The embedding as a list of floats.
        """
        # Wrap the single input in a list and generate embeddings
        embeddings_tensor = self._multimodal_embedding_function([input_item])
        return embeddings_tensor.tolist()[0]

    def _multimodal_embedding_function(self, inputs):
        """
        Generate embeddings for text and/or image inputs using a multimodal model.

        Args:
            inputs (list): A list of inputs, where each input is either:
                - A string (text)
                - A PIL.Image object (image)

        Returns:
            torch.Tensor: A tensor of embeddings with shape (num_inputs, embedding_dim).
        """
        # Separate text and image inputs
        texts = [item for item in inputs if isinstance(item, str)]
        images = [item for item in inputs if isinstance(item, Image.Image)]

        # Handle the case where there are no inputs
        if not texts and not images:
            raise Exception

        # Preprocess inputs conditionally
        embeddings = []
        if "openai/clip-vit" in self.model_name:
            with torch.no_grad():
                if not texts:
                    inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
                    outputs_image = self.model.visual_projection(self.model.vision_model(**inputs).pooler_output)
                if not images:
                    inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
                    outputs_text = self.model.text_projection(self.model.text_model(**inputs).pooler_output)

                if texts:
                    embeddings.append(outputs_text.cpu().numpy())
                if images:
                    embeddings.append(outputs_image.cpu().numpy())
        else:
            if not texts:
                embeddings.append(self.model.encode_image(images))
            if not images:
                embeddings.append(self.model.encode_text(texts))

        # Concatenate embeddings into a single tensor
        embeddings = np.concat(embeddings, axis=0)

        return embeddings


class PaperProcessor:
    def __init__(self, summary: bool = True):
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.data_folder = data_folder
        self.vectorstore = None
        self.rags = {}
        self.retrievers = {}
        self.embedding_function = MultimodalEmbeddingFunction(model_name="jinaai/jina-clip-v1")
        logger.info("Loading pretrained summarizer")
        self.summarizer = pipeline("summarization", model="Falconsai/text_summarization", device=self.device)
        self.llm = CustomLLM()
        self.summary = summary
        self.data = {}

    def create_first_page_summary(self, text: str):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": f"create a summary for RAG. if available, include"
                                                                  f": names of the authors, name of the paper (title),"
                                                                  f" outline key ideas\n\n"
                                                                  f"Use this page: {text}"}]}
        ]

        processed = self.llm.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = {k: v.to(self.device) for k, v in processed.items()}
        generated_ids = self.llm.model.generate(**inputs, max_new_tokens=128)
        prompt_len = len(processed['input_ids'][0])
        mod_gen_id = generated_ids[0][prompt_len:].reshape(1, -1)
        generated_texts = self.llm.processor.batch_decode(
            mod_gen_id,
            skip_special_tokens=True,
        )
        return generated_texts[0].strip()

    def __setup_vectorstore(self, name: str):
        collection = Chroma(
            collection_name=name,
            embedding_function=self.embedding_function
        )
        return collection

    async def process(self, paper_name: str = "2307.00651v1"):
        if paper_name in self.rags:
            return

        vectorstore = self.__setup_vectorstore(paper_name)

        available_papers = os.listdir(self.data_folder)
        if paper_name not in available_papers:
            self.download_paper(paper_name)

        # extract
        logger.info(f"PDF elements extraction {paper_name}")
        extracted_files_path = self.data_folder / f"{paper_name}"
        raw_pdf_elements = extract_pdf_elements(str(extracted_files_path), paper_name)
        self.data["raw_pdf_elements"] = raw_pdf_elements
        logger.debug(f"PDF elements extracted. Raw elements: {len(raw_pdf_elements)}")
        texts, tables = categorize_elements(raw_pdf_elements)
        logger.debug(f"PDF elements extracted. Text: {len(texts)}. Tables: {len(tables)}")

        # Создаем объект CharacterTextSplitter для разбиения текста на части (чанки)
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000,
            chunk_overlap=500
        )
        joined_texts = " ".join(texts)
        texts_token = text_splitter.split_text(joined_texts)
        self.data["texts_token"] = texts_token

        # summaries
        texts_summaries = None
        if self.summary:
            first_page_summary = self.create_first_page_summary(texts_token[0])
            logger.debug(f"First page: {texts_token[0]}")
            logger.debug(f"First page summary: {first_page_summary}")
            texts_summaries = self.summarizer(
                texts_token,  # List of input texts
                max_length=500,  # Maximum length of the summary
                min_length=25,  # Minimum length of the summary
                do_sample=False,  # Use deterministic summarization
                batch_size=8  # Number of texts processed in parallel
            )
            # summarized = self.summarizer(tt, max_length=500, min_length=30, do_sample=False)[0]["summary_text"]
            texts_summaries = [i["summary_text"] for i in texts_summaries]
            texts_summaries.append(first_page_summary)
            self.data["texts_summaries"] = texts_summaries

        # setup retriever
        retriever = self.create_multi_vector_retriever(
            vectorstore=vectorstore,
            text_summaries=texts_summaries,
            texts=texts_token,
            table_summaries=None,
            tables=tables,
            image_summaries=None,
            images=None,
        )
        self.retrievers[paper_name] = retriever
        rag = self.multi_modal_rag_chain(retriever)
        self.rags[paper_name] = rag

    def download_paper(self, arxiv_id: str = "2307.00651"):
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        response = requests.get(pdf_url)
        output_path = Path(self.data_folder, arxiv_id, f"{arxiv_id}.pdf")
        extracted_files_path = self.data_folder / f"{arxiv_id}"
        if not os.path.exists(extracted_files_path):
            os.makedirs(extracted_files_path)
        with open(output_path, "wb") as f:
            f.write(response.content)

        logger.info(f"PDF saved to {output_path}")

    @staticmethod
    def split_image_text_types(docs):
        """
        Разделяет документы на изображения и текстовые данные.

        Аргументы:
        docs: Список документов, содержащих изображения (в формате base64) и текст.

        Возвращает:
        Словарь с двумя списками: изображения и тексты.
        """
        b64_images = []
        texts = []
        for doc in docs:
            if isinstance(doc, Document):
                doc = doc.page_content
            if looks_like_base64(doc) and is_image_data(doc):
                doc = resize_base64_image(doc, size=(1300, 600))
                b64_images.append(doc)
            else:
                texts.append(doc)
        logger.debug(f"Found texts: {len(texts)}")
        return {"images": b64_images, "texts": texts}

    @staticmethod
    def create_multi_vector_retriever(
            vectorstore,
            text_summaries: list | None = None,
            texts: list | None = None,
            table_summaries: list | None = None,
            tables: list | None = None,
            image_summaries: list | None = None,
            images: list | None = None
    ):
        """
        Функция для создания ретривера, который может извлекать данные из разных источников.

        Аргументы:
        vectorstore: Векторное хранилище для хранения векторных представлений документов.
        text_summaries: Список суммаризаций текстовых элементов.
        texts: Список исходных текстов.
        table_summaries: Список суммаризаций таблиц.
        tables: Список исходных таблиц.
        image_summaries: Список суммаризаций изображений.
        images: Список изображений в формате base64.

        Возвращает:
        Созданный ретривер, который может извлекать данные из различных источников.
        """

        # Создаем хранилище для метаданных документов в памяти
        store = InMemoryStore()
        id_key = "doc_id"  # Ключ для идентификации документов в хранилище

        # Создаем многофакторный ритривер
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=store,
            id_key=id_key,
        )

        # Функция добавления документов в ритривер
        def add_documents(retriever, doc_summaries, doc_contents):
            """
            Функция для добавления документов и их метаданных в ритривер.

            Аргументы:
            retriever: Ретривер, в который будут добавляться документы.
            doc_summaries: Список суммаризаций документов.
            doc_contents: Список исходных содержимых документов.
            """
            # Генерируем уникальные идентификаторы для каждого документа
            doc_ids = [str(uuid.uuid4()) for _ in doc_summaries]

            # Создаем документы для векторного хранилища из суммаризаций
            if doc_summaries:
                logger.debug(f"Adding summaries: {doc_summaries[0]}")
                docs = [
                    Document(page_content=s, metadata={id_key: doc_ids[i]})
                    for i, s in enumerate(doc_summaries)
                ]
            else:
                logger.debug("Adding original texts")
                docs = [
                    Document(page_content=s, metadata={id_key: doc_ids[i]})
                    for i, s in enumerate(doc_contents)
                ]

            # Добавляем документы в векторное хранилище
            # retriever.vectorstore.add_documents(summary_docs)
            logger.debug(f"Docs: {len(docs)}")
            retriever.vectorstore.add_documents(docs)

            # Добавляем метаданные документов в хранилище
            retriever.docstore.mset(list(zip(doc_ids, doc_summaries)))

        # Добавляем суммаризации текстов и таблиц, если они присутствуют
        if texts:  # text_summaries:
            logger.debug(f"Adding texts: {len(texts)}")
            logger.debug(f"Adding text summaries: {len(text_summaries)}")
            add_documents(retriever, text_summaries, texts)
        if tables:
            add_documents(retriever, table_summaries, tables)
        if images:
            add_documents(retriever, image_summaries, images)

        return retriever  # Возвращаем созданный ритривер

    @staticmethod
    def debug_context(input_data):
        """
        Debugging function to inspect the context passed to the model.
        """
        context = input_data["context"]
        question = input_data["question"]
        print("Question:", question)
        print("Retrieved Context:")
        for i, doc in enumerate(context):
            if doc == "texts":
                for i, d in enumerate(context["texts"]):
                    print(f"Document {i + 1}:")
                    print(f"Content: {d}")
                    print("-" * 50)
        return input_data

    def multi_modal_rag_chain(self, retriever):
        """
        Создает RAG цепочку для работы с мультимодальными запросами, включая текст и изображения.

        Аргументы:
        retriever: Ритривер для получения данных.

        Возвращает:
        Цепочка для обработки запросов с учетом текста и изображений.
        """

        # Определяем цепочку обработки запросов
        chain = (
                {
                    "context": retriever | RunnableLambda(self.split_image_text_types),
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(self.debug_context)
                | RunnableLambda(self.llm.generate)
                | StrOutputParser()
        )

        return chain

    def respond(self, question: str, paper_id: str):
        chain = self.rags[paper_id]
        return chain.invoke(question)


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
        filename=Path(path, fname + ".pdf"),
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3000,
        combine_text_under_n_chars=2000,
        analyzed_image_output_dir_path=Path(path),
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
    logger.info("Starting categorizing elements...")
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


if __name__ == "__main__":
    test_multimodal = False
    if test_multimodal:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)

        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

        inputs = processor(text=["a photo of a cat", "a photo of a dog"], return_tensors="pt", padding=True).to(device)

        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

        print(probs)

    pp = PaperProcessor()
    pp.process()
    print(pp.respond("tell me about the abstract?", "2307.00651v1"))
