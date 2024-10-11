class BaseQuantizer(object):
    def __init__(self, quant_hub_linear, w_bits=16, a_bits=16, offload=None, device=None, a_qtype="per_tensor", w_qtype="per_group", w_has_zero=False, a_has_zero=False, w_unsign=True, a_unsign=True, quantization_type='static'):
        self.wbit = w_bits
        self.abit = a_bits
        self.quant_hub_linear = quant_hub_linear
        self.offload = offload
        self.device = device
        self.a_qtype = a_qtype
        self.w_qtype = w_qtype
        self.w_has_zero = w_has_zero
        self.a_has_zero = a_has_zero
        self.w_unsign = w_unsign
        self.a_unsign = a_unsign 
        self.quantization_type = quantization_type
        
    def add_hook(self):
        pass
        
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def quantize(self):
        pass

    def cpu(self):
        return self.to('cpu')

    def gpu(self):
        return self.to('cuda')

    def cuda(self, device=None):
        return self.to('cuda')

