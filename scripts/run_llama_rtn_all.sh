python examples/quantize_eval.py --model {path_to_model} --quant-config ./configs/rtn_ceval_hm_quant_config.yaml --eval-ppl --save llama-2-7b-quant.pth
python examples/quantize_eval.py --model {path_to_model} --quant-config ./configs/rtn_ceval_ss_quant_config.yaml --eval-ppl --save llama-2-7b-quant.pth
python examples/quantize_eval.py --model {path_to_model} --quant-config ./configs/rtn_ceval_st_quant_config.yaml --eval-ppl --save llama-2-7b-quant.pth
python examples/quantize_eval.py --model {path_to_model} --quant-config ./configs/rtn_cmmlu_all_quant_config.yaml --eval-ppl --save llama-2-7b-quant.pth
python examples/quantize_eval.py --model {path_to_model} --quant-config ./configs/rtn_cmmlu_hm_quant_config.yaml --eval-ppl --save llama-2-7b-quant.pth
python examples/quantize_eval.py --model {path_to_model} --quant-config ./configs/rtn_cmmlu_ss_quant_config.yaml --eval-ppl --save llama-2-7b-quant.pth
python examples/quantize_eval.py --model {path_to_model} --quant-config ./configs/rtn_cmmlu_st_quant_config.yaml --eval-ppl --save llama-2-7b-quant.pth