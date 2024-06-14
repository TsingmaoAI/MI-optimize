import yaml
import logging
from . import STR_TO_PRECISION
from mi_optimize.datasets.data_loader import get_calibrate_dataset
import sys
sys.path.append("../..")

# def quant
def quantize(model, tokenizer, quant_config):
    with open('configs/default_quant_config.yaml', 'r') as file:
        default_quant_config = yaml.safe_load(file)
    
    algo = quant_config['algo']
    
    config = default_quant_config[algo].copy()
    config['kwargs'].update(quant_config['kwargs'])
    config['calibrate_name']=quant_config['calibrate_name']
        
    logging.info(f"Quantization algorithm: {algo}")
    
    kwargs = config['kwargs']
    kwargs['abit'] = STR_TO_PRECISION[kwargs.pop('a_dtype')]
    kwargs['wbit'] = STR_TO_PRECISION[kwargs.pop('w_dtype')]
    logging.info(f"Quantization kwargs: {kwargs}")
    
    model_type = kwargs.get('model_type')
    
    calibrate_name = config['calibrate_name']
    logging.info(f"calibrate_name: {calibrate_name}")
    
    calibrate_data = get_calibrate_dataset(calibrate_name=calibrate_name, tokenizer=tokenizer, nsamples=kwargs['num_calibrate'], seqlen=kwargs['calibrate_seq_length'])
    
    if model_type == 'llama':
        from mi_optimize.quantization.models.llama_seq import llama_sequential
        model = llama_sequential(model, algo, calibrate_data, **kwargs)
        return model
    elif model_type == 'baichuan':
        from mi_optimize.quantization.models.baichuan_seq import baichuan_sequential
        model = baichuan_sequential(model, algo, calibrate_data, **kwargs)
        return model
    elif model_type == 'chatglm':
        from mi_optimize.quantization.models.baichuan_seq import chatglm_sequential
        model = chatglm_sequential(model, algo, calibrate_data, **kwargs)
        return model
    elif model_type == 'other_model':
        from mi_optimize.quantization.models.quant_other_model import quant_your_model
        model = quant_your_model(model, algo, calibrate_data, **kwargs)
        return model
    else:
        raise ValueError(f'not support {model_type}')



