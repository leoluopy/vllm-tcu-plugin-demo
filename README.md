# vllm-tcu-plugin-demo
vLLM plugin for tcu (tcu is hardware backend name) for beginners to add a plugin to vllm 

# For more info see Blog 
https://www.zhihu.com/column/c_1916902881216399104 
+ main arch and code Description
+ attention / platform / custom op / worker / distributed communicator 

# HOW TO 
 ```
 1. make sure torch_tcu and vllm_tcu is uninstalled
 ```

```python
pip uninstall torch_tcu vllm_tcu
rm -rf /usr/local/lib/python3.12/dist-packages/torch_tcu*
rm -rf /usr/local/lib/python3.12/dist-packages/vllm_tcu*
```

 ```
 2. install torch tcu backend 
 ```

```python
cd torch_tcu
python setup.py install 
```

 ```
 3. run torch tcu unit tests
 ```

```python
python ut_base_op.py
python ut_linear.py
python ut_set_device_idx.py
python ut_Attension_newBackEnd.py
python ut_load_callbackend_modelR50.py
```

 ```
4. install vllm-tcu  (using develop mode ) 
 ```

```python
pip install vllm==0.8.5  -i https://pypi.doubanio.com/simple/
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu

python setup.py develop 
```

```
5. run vllm-tcu unit test , NOTE: make sure change the config of your LLM
```

```python
python test_plugins.py 

expected output:
🔄 注册自定义模型
通用插件 register_custom_models 返回: ['tcu-llama', 'tcu-gpt']
✅ 已注册 TCU Platform 平台插件
平台插件 register_platform_plugins 返回: vllm_tcu.platform.TCUPlatform


python ut_single_vllm_layer_tcu_2token.py
expected output:
see log.txt in main repo directory 
```