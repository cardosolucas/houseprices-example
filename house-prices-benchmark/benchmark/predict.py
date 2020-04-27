import numpy as np

from tools.utils import get_time, calculate_time
from tools.log import get_logger

logger = get_logger('predict')

def format_input(message):
    initial_time = get_time()
    logger.info('Formatting input...')
    np_array = np.fromstring(message, dtype=float, sep=',')
    np_array = np_array.reshape(1, -1)
    logger.info('Formatting input end: ' + str(calculate_time(initial_time)) + ' seconds.')
    return np_array 

def predict(model, message):
    return model.predict(format_input(message))[0]