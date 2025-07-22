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

