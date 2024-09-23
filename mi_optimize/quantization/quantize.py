import logging
from . import STR_TO_PRECISION
from mi_optimize.datasets.data_loader import get_calibrate_loader
from mi_optimize.quantization.models import llama_sequential, baichuan_sequential, chatglm_sequential, quant_other_model

def quantize(model, tokenizer, quant_config):
    kwargs = quant_config['kwargs']
    kwargs['abit'] = STR_TO_PRECISION[kwargs.pop('a_dtype')] 
    kwargs['wbit'] = STR_TO_PRECISION[kwargs.pop('w_dtype')]
    calibrate_config = quant_config['calibrate_config']
    algo = quant_config['algo']
    model_type = getattr(model.config, 'model_type', 'other_model')

    # Logging details about the quantization process
    logging.info(f"Quantization algorithm: {algo}")
    logging.info(f"Quantization kwargs: {kwargs}")
    logging.info(f"Model type: {model_type}")
    logging.info(f"Calibrate config: {calibrate_config}")

    # Get the calibrate loader based on the tokenizer and calibration configuration
    calibrate_loader = get_calibrate_loader(tokenizer, calibrate_config)

    # Mapping model types to their respective quantization functions
    if kwargs['abit'] <=8 or kwargs['wbit']<=8:
        if model_type == 'llama':
            return llama_sequential(model, algo, calibrate_loader, **kwargs)

        if model_type == 'baichuan':
            return baichuan_sequential(model, algo, calibrate_loader, **kwargs)

        if model_type == 'chatglm':
            return chatglm_sequential(model, algo, calibrate_loader, **kwargs)

        else:
            return quant_other_model(model, algo, calibrate_loader, **kwargs)
    else:
        return model

