import unittest
import torch
import yaml
import argparse
from transformers import AutoModelForCausalLM, LlamaTokenizer
from mi_optimize import quantize
from mi_optimize.export import export_module

model = 'meta-llama/Llama-2-7b-hf'
quant_config = './configs/smoothquant_config.yaml'
        
class TestModelConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Assuming the quantized model and tokenizer are already saved
        cls.tokenizer = LlamaTokenizer.from_pretrained(model, trust_remote_code=True)
        cls.quant_model = 'llama-2-7b-quant.pth'
        cls.input_text = "Llama is a large language model"
        cls.input_ids = cls.tokenizer.encode(cls.input_text, return_tensors="pt").cuda()

    def test_output_consistency(self):
        # Re-quantize and test the output again
        model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True).half().cuda()
        with open(quant_config, 'r') as file:
            config = yaml.safe_load(file)
        quant_model = quantize(model=model, tokenizer=self.tokenizer, quant_config=config['quant_config'])
        output = quant_model.generate(self.input_ids, do_sample=False)
        decoded_output_quantized = self.tokenizer.decode(output[0], skip_special_tokens=True)
        print('quantized outputs: ', decoded_output_quantized)
        quant_model = export_module(quant_model)
        torch.save(quant_model, self.quant_model)
        
        # Generate output from the loaded quantized model
        model = torch.load(self.quant_model).cuda()
        output = model.generate(self.input_ids, do_sample=False)
        decoded_output_loaded = self.tokenizer.decode(output[0], skip_special_tokens=True)
        print('load model outputs: ', decoded_output_loaded)
        
        # Assert the outputs are the same
        self.assertEqual(decoded_output_loaded, decoded_output_quantized)

if __name__ == '__main__':
    unittest.main()