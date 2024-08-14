ChatGLMForConditionalGeneration(
  (transformer): ChatGLMModel(
    (embedding): Embedding(
      (word_embeddings): Embedding(65024, 4096)
    )
    (rotary_pos_emb): RotaryEmbedding()
    (encoder): GLMTransformer(
      (layers): ModuleList(
        (0-27): 28 x GLMBlock(
          (input_layernorm): RMSNorm()
          (self_attention): SelfAttention(
            (query_key_value): LinearQuantHub(
              (core): Linear(in_features=4096, out_features=4608, bias=True)
            )
            (core_attention): CoreAttention(
              (attention_dropout): Dropout(p=0.0, inplace=False)
            )
            (dense): LinearQuantHub(
              (core): Linear(in_features=4096, out_features=4096, bias=False)
            )
          )
          (post_attention_layernorm): RMSNorm()
          (mlp): MLP(
            (dense_h_to_4h): LinearQuantHub(
              (core): Linear(in_features=4096, out_features=27392, bias=False)
            )
            (dense_4h_to_h): LinearQuantHub(
              (core): Linear(in_features=13696, out_features=4096, bias=False)
            )
          )
        )
      )
      (final_layernorm): RMSNorm()
    )
    (output_layer): LinearQuantHub(
      (core): Linear(in_features=4096, out_features=65024, bias=False)
    )
  )
)



#量化后的chatglm模型
ChatGLMForConditionalGeneration(
  (transformer): ChatGLMModel(
    (embedding): Embedding(
      (word_embeddings): Embedding(65024, 4096)
    )
    (rotary_pos_emb): RotaryEmbedding()
    (encoder): GLMTransformer(
      (layers): ModuleList(
        (0-27): 28 x GLMBlock(
          (input_layernorm): RMSNorm()
          (self_attention): SelfAttention(
            (query_key_value): QLinear()
            (core_attention): CoreAttention(
              (attention_dropout): Dropout(p=0.0, inplace=False)
            )
            (dense): QLinear()
          )
          (post_attention_layernorm): RMSNorm()
          (mlp): MLP(
            (dense_h_to_4h): QLinear()
            (dense_4h_to_h): QLinear()
          )
        )
      )
      (final_layernorm): RMSNorm()
    )
    (output_layer): LinearQuantHub(
      (core): Linear(in_features=4096, out_features=65024, bias=False)
    )
  )
)

['transformer.embedding.word_embeddings.weight', 
 'transformer.rotary_pos_emb.inv_freq', 
 'transformer.encoder.layers.0.input_layernorm.weight', 
 'transformer.encoder.layers.0.self_attention.query_key_value.bias', 
 'transformer.encoder.layers.0.self_attention.query_key_value.w_scale', 
 'transformer.encoder.layers.0.self_attention.query_key_value.w_zero_point', 
 'transformer.encoder.layers.0.self_attention.query_key_value.pack_weight', 
 'transformer.encoder.layers.0.self_attention.dense.bias', 
 'transformer.encoder.layers.0.self_attention.dense.w_scale', 
 'transformer.encoder.layers.0.self_attention.dense.w_zero_point', 
 'transformer.encoder.layers.0.self_attention.dense.pack_weight', 
 'transformer.encoder.layers.0.post_attention_layernorm.weight', 
 'transformer.encoder.layers.0.mlp.dense_h_to_4h.bias', 
 'transformer.encoder.layers.0.mlp.dense_h_to_4h.w_scale', 
 'transformer.encoder.layers.0.mlp.dense_h_to_4h.w_zero_point', 
 'transformer.encoder.layers.0.mlp.dense_h_to_4h.pack_weight', 
 'transformer.encoder.layers.0.mlp.dense_4h_to_h.bias', 
 'transformer.encoder.layers.0.mlp.dense_4h_to_h.w_scale', 
 'transformer.encoder.layers.0.mlp.dense_4h_to_h.w_zero_point', 
 'transformer.encoder.layers.0.mlp.dense_4h_to_h.pack_weight', 
 'transformer.encoder.final_layernorm.weight', 'transformer.output_layer.core.weight']

#awq
['transformer.embedding.word_embeddings.weight',
  'transformer.rotary_pos_emb.inv_freq', 
  'transformer.encoder.layers.0.input_layernorm.weight', 
  'transformer.encoder.layers.0.self_attention.query_key_value.bias', 
  'transformer.encoder.layers.0.self_attention.query_key_value.w_scale',
  'transformer.encoder.layers.0.self_attention.query_key_value.w_zero_point',
  'transformer.encoder.layers.0.self_attention.query_key_value.pack_weight', 
  'transformer.encoder.layers.0.self_attention.dense.w_scale', 
  'transformer.encoder.layers.0.self_attention.dense.w_zero_point', 
  'transformer.encoder.layers.0.self_attention.dense.pack_weight', 
  'transformer.encoder.layers.0.post_attention_layernorm.weight', 
  'transformer.encoder.layers.0.mlp.dense_h_to_4h.w_scale', 
  'transformer.encoder.layers.0.mlp.dense_h_to_4h.w_zero_point', 
  'transformer.encoder.layers.0.mlp.dense_h_to_4h.pack_weight', 
  'transformer.encoder.layers.0.mlp.dense_4h_to_h.w_scale', 
  'transformer.encoder.layers.0.mlp.dense_4h_to_h.w_zero_point', 
  'transformer.encoder.layers.0.mlp.dense_4h_to_h.pack_weight']