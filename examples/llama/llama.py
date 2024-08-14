LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(32000, 4096)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaSdpaAttention(
          (q_proj): LinearQuantHub(
            (core): Linear(in_features=4096, out_features=4096, bias=False)
          )
          (k_proj): LinearQuantHub(
            (core): Linear(in_features=4096, out_features=4096, bias=False)
          )
          (v_proj): LinearQuantHub(
            (core): Linear(in_features=4096, out_features=4096, bias=False)
          )
          (o_proj): LinearQuantHub(
            (core): Linear(in_features=4096, out_features=4096, bias=False)
          )
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): LinearQuantHub(
            (core): Linear(in_features=4096, out_features=11008, bias=False)
          )
          (up_proj): LinearQuantHub(
            (core): Linear(in_features=4096, out_features=11008, bias=False)
          )
          (down_proj): LinearQuantHub(
            (core): Linear(in_features=11008, out_features=4096, bias=False)
          )
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
  )
  (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
)

LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(32000, 4096)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): QLinear()
          (k_proj): QLinear()
          (v_proj): QLinear()
          (o_proj): QLinear()
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): QLinear()
          (up_proj): QLinear()
          (down_proj): QLinear()
          (act_fn): SiLUActivation()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
  )
  (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
)

['model.embed_tokens.weight', 
 'model.layers.0.self_attn.q_proj.w_scale', 
 'model.layers.0.self_attn.q_proj.w_zero_point', 
 'model.layers.0.self_attn.q_proj.pack_weight', 
 'model.layers.0.self_attn.k_proj.w_scale', 
 'model.layers.0.self_attn.k_proj.w_zero_point',
 'model.layers.0.self_attn.k_proj.pack_weight', 
 'model.layers.0.self_attn.v_proj.w_scale', 
 'model.layers.0.self_attn.v_proj.w_zero_point', 
 'model.layers.0.self_attn.v_proj.pack_weight', 
 'model.layers.0.self_attn.o_proj.w_scale', 
 'model.layers.0.self_attn.o_proj.w_zero_point', 
 'model.layers.0.self_attn.o_proj.pack_weight', 
 'model.layers.0.mlp.gate_proj.w_scale', 
 'model.layers.0.mlp.gate_proj.w_zero_point', 
 'model.layers.0.mlp.gate_proj.pack_weight', 
 'model.layers.0.mlp.up_proj.w_scale', 
 'model.layers.0.mlp.up_proj.w_zero_point', 
 'model.layers.0.mlp.up_proj.pack_weight', 
 'model.layers.0.mlp.down_proj.w_scale', 
 'model.layers.0.mlp.down_proj.w_zero_point', 
 'model.layers.0.mlp.down_proj.pack_weight', 
 'model.layers.0.input_layernorm.weight', 
 'model.layers.0.post_attention_layernorm.weight'
 'model.norm.weight', 
 'lm_head.weight'
]
