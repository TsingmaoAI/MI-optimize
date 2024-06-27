# python examples/quantize_eval.py --model /home/wf/models/Llama2-Chinese-13b-Chat --quant-config ./configs/gptq_cmmlu_hm_quant_config.yaml --eval-ppl --eval-ceval --eval-cmmlu --eval-boss
# python examples/quantize_eval.py --model /home/wf/models/Llama2-Chinese-13b-Chat --quant-config ./configs/gptq_cmmlu_ss_quant_config.yaml --eval-ppl --eval-ceval --eval-cmmlu --eval-boss
# python examples/quantize_eval.py --model /home/wf/models/Llama2-Chinese-13b-Chat --quant-config ./configs/gptq_cmmlu_st_quant_config.yaml --eval-ppl --eval-ceval --eval-cmmlu --eval-boss

python examples/quantize_eval.py --model /home/wf/models/Llama2-Chinese-13b-Chat --quant-config ./configs/gptq_cmmlu_hm_quant_config.yaml --save 'llama2-13b-chat-gptq-4bit.pth'