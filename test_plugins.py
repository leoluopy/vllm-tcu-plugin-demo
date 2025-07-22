

from importlib.metadata import entry_points

def test_plugins():
    """测试所有注册的entrypoints"""
    # 测试通用插件
    general_eps = entry_points().select(group="vllm.general_plugins")
    for ep in general_eps:
        func = ep.load()
        result = func()
        print(f"通用插件 {ep.name} 返回: {result}")

    # 测试平台插件
    platform_eps = entry_points().select(group="vllm.platform_plugins")
    for ep in platform_eps:
        func = ep.load()
        result = func()
        print(f"平台插件 {ep.name} 返回: {result}")

if __name__ == "__main__":
    test_plugins()

# expected output
# 🔄 注册自定义模型
# 通用插件 register_custom_models 返回: ['gcu-llama', 'gcu-gpt']
# ✅ 已注册 GCU 平台插件
# 平台插件 register_platform_plugins 返回: {'platform': 'gcu'}

