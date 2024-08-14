"""

Date:8/05/2023
author:niexin
description:将llama导出为.flm文件,而后利用fastllm框架进行推理

"""
import struct
import builtins, os, json
import numpy as np
import torch
from transformers import PreTrainedTokenizerFast,LlamaTokenizer,LlamaForCausalLM
from mi_optimize.export.qnn import QLinear
from tokenizers.decoders import ByteLevel

def writeString(fo, s):
    bytes = s.encode()
    fo.write(struct.pack('i', len(bytes)))
    fo.write(bytes)

def writeKeyValue(fo, key, value):
    writeString(fo, key)
    writeString(fo, value)

quantization_method_set =[
    "gptq",
    "rtn",
    "awq",
    "smoothqunat"
]

fastllm_data_type_dict = {
    "float32": 0,
    "bfloat16": 1,
    "int16" : 2,
    "int8": 3,
    "int4": 4,
    "int2": 5,
    "bit" : 6,
    "float16" :7,
    "int4_nozero":8,
    "int4g": 9,
}
fastllm_weight_type_dict = {
    "QLinear": 1,
    "embedding": 2,
}

def write_int8(fo,pack_weight,w_scale,w_zero_point):
    fo.write(struct.pack('i', len(pack_weight.shape)))  #写入权重的形状的长度，可以得知权重的维度数量并写入flm文件

    #由于pack_weight的形状是（out,in）类型的，而fastllm输入的矩阵需要（in,out）形状的
    #TODO，pack_weight的数据类型是int32,这里假定pack_weight.shape[0]*4就是原来真实的维度，后续会作优化
    fo.write(struct.pack('i', pack_weight.shape[1]))
    fo.write(struct.pack('i', pack_weight.shape[0]*4))

    fo.write(struct.pack('i', 3))  #3代表为量化成int8，可参看fastllm_data_type_dict字典
    fo.write(struct.pack('i', 0))  #这里是为了指定量化沿着那个维度展开，0代表per-channel之类的
    # 写入量化的scale和零点
    for i in range(pack_weight.shape[1]):
        fo.write(struct.pack('f', w_scale[i]))
        fo.write(struct.pack('f', w_zero_point[i]))

    v = np.zeros((pack_weight.shape[0]*4,pack_weight.shape[1]),dtype=np.int32)
    #解压pack_weight
    for i  in range(pack_weight.shape[0]*4):
        row = i//4
        off = i%4
        v[i] = (pack_weight[row]>>(32-(off+1)*8))&(0x000000FF)
    #转为uint8
    v = v.astype(np.uint8)
    #转置
    v = np.transpose(v)
    v = np.ascontiguousarray(v)

    #写入uint8权重
    fo.write(v.data)


def write_int4g(fo, v, groupCnt = -1):
    if (groupCnt == -1):
        groupCnt = 128
    k = v.shape[0]
    m = v.shape[1]
    group = (m - 1) // groupCnt + 1
    pad = group * groupCnt - m
    if (pad > 0):
        v = np.concatenate((v, np.zeros([k, pad])), axis = 1)
    v.resize(k * group, groupCnt)
    c_min = np.expand_dims(v.min(axis = -1), -1)
    c_max = np.expand_dims(v.max(axis = -1), -1)
    c_scale = (c_max - c_min) / 15.0
    c_zero = np.round(0.0 - c_min / c_scale)
    c_zero = c_zero.clip(0, 15)
    c_min = -c_scale * c_zero

    v = (v - c_min) / c_scale
    v = (v + 0.5).astype(np.int8).clip(0, 15).astype(np.uint8)

    if (pad > 0):
        v.resize(k, group * groupCnt)
        v = v[:, :-pad].copy(order = 'C')

    v = v[:, 0::2] * 16 + v[:, 1::2]
    fo.write(struct.pack('i', 9))
    fo.write(struct.pack('i', 0))
    fo.write(struct.pack('i', group))
    fo.write(struct.pack('i', groupCnt))
    for i in range(c_min.shape[0]):
        fo.write(struct.pack('f', c_min[i][0]));
        fo.write(struct.pack('f', c_max[i][0]));
    fo.write(v)

def write_int4(fo,pack_weight,w_scale,w_zero_point):
    fo.write(struct.pack('i', len(pack_weight.shape)))  
    fo.write(struct.pack('i', pack_weight.shape[1]))
    fo.write(struct.pack('i', pack_weight.shape[0]*8))
    fo.write(struct.pack('i', 4))  
    fo.write(struct.pack('i', 0))
    for i in range(pack_weight.shape[1]):
        fo.write(struct.pack('f', w_scale[i]))
        fo.write(struct.pack('f', w_zero_point[i]))
    v = np.zeros((pack_weight.shape[0]*8,pack_weight.shape[1]),dtype=np.int32)
    for i  in range(pack_weight.shape[0]*8):
        row = i//8
        off = i%8
        v[i] = (pack_weight[row]>>(32-(off+1)*4))&(0x0000000F)
    v = v.astype(np.uint8)
    v = np.transpose(v)
    v = np.ascontiguousarray(v)
    v = v[:, 0::2] * 16 + v[:, 1::2]
    fo.write(v.data)


def factor_fuse(model , quantization_method):
    
    if quantization_method == 'rtn' or quantization_method == 'gptq': #如果是rtn和gptq，则不需要做参数融合
        return
    
    if quantization_method == 'awq':
        for layer in model.transformer.encoder.layers:
            block = layer
            
            #将qkv的factor融入到前一层的RMSnorm的权重中
            block.input_layernorm.weight = block.input_layernorm.weight.div(block.self_attention.query_key_value.smooth_factor.view(1, -1))

            #将dense的facotr融入到前面的qkv矩阵当中的v矩阵的scale中。


                

def llama2flm(
    exportPath,
    model,          #假定现在输入的模型都是经过MI-optimize量化打包过的
    tokenizer = None,
    pre_prompt = None,
    user_role = None,
    bot_role = None,
    history_sep = None,
    eos_id = None,
    dtype = "float16", #dtype代表权重的量数据类型
    quantization_method = 'rtn',
        ):
    
    int4g_groupcnt = -1
    if (dtype.startswith("int4g") and len(dtype) > 5):
        try:
            int4g_groupcnt = int(dtype[5:])
            dtype = "int4g";
        except:
            print("dtype should be like \"int4g256\"")
            exit(0)
    if (dtype not in fastllm_data_type_dict):
        print("dtype should be one of ", list(fastllm_data_type_dict.keys()))
        exit(0)

    
    # 0.1 model info
    modelInfo = model.config.__dict__
    if model.generation_config is not None:
        modelInfo.update(model.generation_config.__dict__)
    if ("model_type" not in modelInfo):
        print("unknown model_type.")
        exit(0)

    fo = open(exportPath, "wb")

    # 0. version id
    fo.write(struct.pack('i', 2)) #truct.pack('i', 2) 将整数 2 打包成一个二进制格式的数据。这里的 'i' 表示打包的是一个标准的 4 字节（32 位）整数

    if (pre_prompt is not None):
        modelInfo["pre_prompt"] = pre_prompt
    if (user_role is not None):
        modelInfo["user_role"] = user_role
    if (bot_role is not None):
        modelInfo["bot_role"] = bot_role
    if (history_sep):
        modelInfo["history_sep"] = history_sep
    

    if (modelInfo["model_type"] == "baichuan"):
        if (hasattr(model, "model") and hasattr(model.model, "get_alibi_mask")):
            # Baichuan / Baichuan2 13B
            modelInfo["use_alibi"] = "1"
        modelInfo["pre_prompt"] = ""
        if (modelInfo["vocab_size"] == 125696):
            # Baichuan 2代
            modelInfo["user_role"] = ("<FLM_FIX_TOKEN_" + str(model.generation_config.user_token_id) + ">") if hasattr(model.generation_config, "user_token_id") else "";
        else:
            # Baichuan-13B-chat
            modelInfo["user_role"] = ("<FLM_FIX_TOKEN_" + str(model.generation_config.user_token_id) + "> ") if hasattr(model.generation_config, "user_token_id") else "";
        modelInfo["bot_role"] = ("<FLM_FIX_TOKEN_" + str(model.generation_config.assistant_token_id) + ">") if hasattr(model.generation_config, "assistant_token_id") else "";
        modelInfo["history_sep"] = ""
    if (modelInfo["model_type"] == "qwen"):
        if modelInfo["chat_format"] == "chatml":
            modelInfo["im_end_id"] = tokenizer.im_end_id
            modelInfo["im_start_id"] = tokenizer.im_start_id
    elif (modelInfo["model_type"] == "qwen2"):
        modelInfo["eos_token_id"] = "151645"
    elif (modelInfo["model_type"] == "internlm"):
        modelInfo["eos_token_id"] = "103028"
        if "rotary" in modelInfo:
            rope_scaling = modelInfo.pop("rotary")
            if isinstance(rope_scaling, builtins.dict):
                modelInfo["rope_scaling.type"] = rope_scaling["type"]
                modelInfo["rope_theta"] = rope_scaling["base"]
    elif (modelInfo["model_type"] == "internlm2"):
        modelInfo["eos_token_id"] = "92542"
    if (modelInfo["model_type"] == "chatglm" and hasattr(tokenizer, "build_chat_input")):
        # chatglm3
        modelInfo["pre_prompt"] = "";
        modelInfo["user_role"] = ("<FLM_FIX_TOKEN_" + str(tokenizer.get_command("<|user|>")) + "> \n");
        modelInfo["bot_role"] = ("<FLM_FIX_TOKEN_" + str(tokenizer.get_command("<|assistant|>")) + ">");
        modelInfo["history_sep"] = "";
    if (modelInfo["model_type"] == "chatglm" and hasattr(tokenizer, "name") and tokenizer.name == "GLM4Tokenizer"):
        # glm-4-chat
        modelInfo["pre_prompt"] = "[gMASK]<sop>";
        modelInfo["user_role"] = ("<FLM_FIX_TOKEN_" + str(tokenizer.convert_tokens_to_ids("<|user|>")) + ">\n");
        modelInfo["bot_role"] = ("<FLM_FIX_TOKEN_" + str(tokenizer.convert_tokens_to_ids("<|assistant|>")) + ">");
        modelInfo["history_sep"] = "";
        modelInfo["eos_token_id"] = "151336"
    if "rope_scaling" in modelInfo and isinstance(modelInfo["rope_scaling"], builtins.dict):
        rope_scaling = modelInfo.pop("rope_scaling")
        modelInfo["rope_scaling.type"] = rope_scaling["type"]
        modelInfo["rope_scaling.factor"] = rope_scaling["factor"]
    if eos_id:
        modelInfo["eos_token_id"] = str(eos_id)

    merges = {}
    if tokenizer:
        modelInfo["tokenizer_use_score"] = "1" # 分词带分数
        if len(tokenizer.all_special_tokens) > 0:
            token_set = set()
            for token in [tokenizer.bos_token, tokenizer.eos_token, tokenizer.unk_token, tokenizer.pad_token]:
                for prompt in [pre_prompt, user_role, bot_role, history_sep]:
                    if prompt and str(token) in prompt:
                        modelInfo["tokenizer_has_special_tokens"] = "1"
                token_set.add(str(token))
            if len(tokenizer.all_special_tokens) > len(token_set):
                modelInfo["tokenizer_has_special_tokens"] = "1"
        if hasattr(tokenizer, "sp_model") or (hasattr(tokenizer, "tokenizer") and hasattr(tokenizer.tokenizer, "sp_model")):
            try:
                import sentencepiece.sentencepiece_model_pb2 as model_pb2
                with open(tokenizer.vocab_file, "rb") as f:
                    sp_model_data = f.read()
                    sp_model_proto = model_pb2.ModelProto.FromString(sp_model_data)
                    modelInfo["tokenizer_add_dummy_prefix"] = sp_model_proto.normalizer_spec.add_dummy_prefix
                    if sp_model_proto.normalizer_spec.remove_extra_whitespaces:
                        modelInfo["tokenizer_remove_extra_whitespaces"] = True
            except:
                pass
        elif isinstance(tokenizer, PreTrainedTokenizerFast):
            modelInfo["tokenizer_add_dummy_prefix"] = False
            tokenizer_file_name = "tokenizer.json"    #TODO：这里强行指定了必须要有tokenizer.json，之后可以做优化。
            tokenizer_file = os.path.join(tokenizer.name_or_path, tokenizer_file_name)
            if os.path.exists(tokenizer_file):
                print(tokenizer_file)
                with open(tokenizer_file, "r", encoding='utf-8') as f:
                    tokenizer_data = json.load(f)
                    if "normalizer" in tokenizer_data and tokenizer_data["normalizer"] and "normalizers" in tokenizer_data["normalizer"]:
                        for normalizer in tokenizer_data["normalizer"]["normalizers"]:
                            if normalizer["type"] == "Prepend" and \
                                    (normalizer["prepend"] == '▁' or normalizer["prepend"] == ' '):
                                modelInfo["tokenizer_add_dummy_prefix"] = True
                    if "merges" in tokenizer_data["model"]:
                        bpe_merges = tokenizer_data["model"]["merges"]
                        bpe_merges = [pair.replace(" ", "") for pair in bpe_merges]
                        merges = builtins.dict(zip(bpe_merges, range(0, -len(bpe_merges), -1)))
            if hasattr(tokenizer, "_tokenizer") and hasattr(tokenizer._tokenizer, "decoder") \
                    and isinstance(tokenizer._tokenizer.decoder, ByteLevel):
                modelInfo["tokenizer_byte_as_char"] = True
        else:
            if hasattr(tokenizer, "byte_encoder") and hasattr(tokenizer, "byte_decoder"):
                modelInfo["tokenizer_byte_as_char"] = True
            if not hasattr(tokenizer, "add_prefix_space") or not getattr(tokenizer, "add_prefix_space", True):
                modelInfo["tokenizer_add_dummy_prefix"] = False

    if hasattr(model, "peft_config"):
        adapter_size = len(model.peft_config)
        modelInfo["peft_size"] = adapter_size

    fo.write(struct.pack('i', len(modelInfo)))
    for it in sorted(modelInfo.keys()):
        writeKeyValue(fo, str(it), str(modelInfo[it]))

    if hasattr(model, "peft_config"):
        for adapter_name in model.peft_config.keys():
            adapter_dict = model.peft_config[adapter_name].__dict__
            writeString(fo, adapter_name)
            fo.write(struct.pack('i', len(adapter_dict)))
            for it in adapter_dict.keys():
                writeKeyValue(fo, str(it), str(adapter_dict[it]))

    weight_type_dict = {}
    model = model.cpu()
    dict = model.state_dict()

    # 1. vocab
    if (tokenizer):
        if (hasattr(tokenizer, "tokenizer")):
            if (str(type(tokenizer.tokenizer)).find("Encoding") == -1):
                tokenizer = tokenizer.tokenizer
        if (hasattr(tokenizer, "sp_model")):
            piece_size = tokenizer.sp_model.piece_size()
            fo.write(struct.pack('i', piece_size))
            for i in range(piece_size):
                s = tokenizer.sp_model.id_to_piece(i).encode()
                fo.write(struct.pack('i', len(s)))
                for c in s:
                    fo.write(struct.pack('i', c))
                fo.write(struct.pack('i', i))
                fo.write(struct.pack('f', float(tokenizer.sp_model.get_score(i))))
        else:
            if hasattr(tokenizer, "bpe_ranks"):
                merges = {("".join(bpe_tokens), token_index) for bpe_tokens, token_index in sorted(tokenizer.bpe_ranks.items(), key=lambda kv: kv[1])}
            vocab = tokenizer.get_vocab()
            fo.write(struct.pack('i', len(vocab)))
            for v in vocab.keys():
                score = merges[v] if v in merges else 1.0
                # if (modelInfo["model_type"] == "moss"):
                #     s = [(ord(c) if c not in tokenizer.byte_decoder else tokenizer.byte_decoder[c]) for c in v]
                if (isinstance(v, str)):
                    s = v.encode()
                else:
                    s = v
                fo.write(struct.pack('i', len(s)))
                for c in s:
                    fo.write(struct.pack('i', c))
                fo.write(struct.pack('i', vocab[v]))
                fo.write(struct.pack('f', score))
        if ("tokenizer_has_special_tokens" in modelInfo):
            all_special_tokens = tokenizer.all_special_tokens
            if hasattr(tokenizer, "added_tokens_decoder"):
                for i in tokenizer.added_tokens_decoder:
                    all_special_tokens.append(str(tokenizer.added_tokens_decoder[i]))
            fo.write(struct.pack('i', len(all_special_tokens)))
            for special_token in all_special_tokens:
                writeString(fo, special_token)
    else:
        fo.write(struct.pack('i', 0))

    module_dict = {}
    for key, m in model.named_modules():
        if (isinstance(m, QLinear)):              #Qlinear是MI-optimize转换的量化层
            weight_type_dict[key + ".weight"] = "QLinear"
            module_dict[key + ".weight"] = m
        if (isinstance(m, torch.nn.Embedding)):
            weight_type_dict[key + ".weight"] = "embedding"

    # 2. weight
    #写入权重，从这里开始更改，TODO:目前只是初步开发阶段，只实现了perchannel的8bit,4bit权重量化
    cnt = 0
    for key in dict:
        key_name_list = key.split('.')
        if key_name_list[-1]  != 'w_scale' and key_name_list[-1] != 'w_zero_point' :
            cnt += 1
    fo.write(struct.pack('i', int(cnt)))  #dict = model.state_dict()

    #根据不同的量化方法做一些参数融合，如awq需要将smooth_factor融合到前一层的权重中,目前处于开发阶段
    #factor_fuse(model,quantization_method)

    tot = 0
    for key in dict:
        weight_name = key 
        weight_name_list = weight_name.split('.')

        if weight_name_list[-1] == 'pack_weight' : 
            tot += 1

            pack_weight= dict[weight_name].numpy()

            weight_name_list[-1] = 'weight'
            weight_name = ".".join(weight_name_list)
            writeString(fo, weight_name)         
            
            weight_name_list[-1] = 'w_scale'
            w_scale_name = ".".join(weight_name_list)

            w_scale = dict[w_scale_name].numpy()

            weight_name_list[-1] = 'w_zero_point'
            w_zero_point_name = ".".join(weight_name_list)

            w_zero_point = dict[w_zero_point_name].numpy()
            
            if dtype == "int8":
                write_int8(fo,pack_weight,w_scale,w_zero_point)
            elif dtype == "int4":
                write_int4(fo,pack_weight,w_scale,w_zero_point)
            else:
                raise TypeError(f"不支持的权重量化类型")
        else :
            #TODO:目前只考虑除了linear层的权重可以是int8，int4类型的，其他都是float32,后续需要作优化
            if weight_name_list[-1] != 'w_scale' and weight_name_list[-1] != 'w_zero_point' :
                tot += 1
                writeString(fo, weight_name)
                cur = dict[key].numpy().astype(np.float32)
                fo.write(struct.pack('i', len(cur.shape)))
                for i in cur.shape:
                    fo.write(struct.pack('i', i))
                fo.write(struct.pack('i', 0))
                fo.write(cur.data)
        print("output (", tot, "/", cnt, end = " )\r") #仅使用换行符可以达到刷新的目的。
    print("\nfinish.")
    fo.close()

if __name__ == '__main__' :
    
    tokenizer = LlamaTokenizer.from_pretrained('/home/wf/models/Llama-2-7b-hf',trust_remote_code = True)  #TODO:这行代码与下面交换会导致报错，因为torch.load和torch.save得具有相同的脚本结构，torch.save如果使用了trust_remote_code = true可能会导致这可能会改变 Python 环境或路径，从而影响模块的导入顺序,所以这里把tokenizer放到前面执行
    model = torch.load('/home/wf/nx/MI-optimize/examples/llama/w8a16gptq.pt')
    #model = LlamaForCausalLM.from_pretrained('/home/wf/models/Llama-2-7b-hf', trust_remote_code = True)
    exportPath = '/home/wf/nx/MI-optimize/examples/llama/llama2-int8.flm'
    # model.eval()
    # model.cuda()
    # prompt = 'why is the sky blue'
    # inputs = tokenizer.encode(prompt, return_tensors="pt")
    # outputs = model.generate(inputs.to('cuda'), max_length=100, num_return_sequences=1)
    # response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # print(response)

    # exit(0)
    llama2flm(exportPath=exportPath,model=model,tokenizer=tokenizer,
             pre_prompt = "<FLM_FIX_TOKEN_1>", 
                     user_role = "", bot_role ="", 
                     history_sep = "",
              dtype = 'int8')

    

    