

def register_platform_plugins():
    """平台插件注册入口"""
    print("✅ 已注册 TCU Platform 平台插件")
    # 这里添加实际的平台注册逻辑
    # return {"platform": "tcu"}
    return "vllm_tcu.platform.TCUPlatform"



