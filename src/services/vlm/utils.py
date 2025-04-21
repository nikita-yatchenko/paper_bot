import logging
import time


def get_generation_time(llm, sampling_params, prompts, logger: logging.Logger):
    """
    Get generation time for llm
    :param llm:
    :param sampling_params:
    :param prompts:
    :return:
    """
    start_time = time.time()
    output = llm.generate(prompts, sampling_params=sampling_params)
    end_time = time.time()
    logger.info(f"Результат: {output[0].outputs[0].text}")
    logger.info(f"Время генерации: {end_time - start_time} секунды.")
