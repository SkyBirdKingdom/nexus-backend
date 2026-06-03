# 项目的唯一启动文件 (调用 uvicorn)
import uvicorn

if __name__ == "__main__":
    """
    【架构解析：应用服务器入口】
    这里是整个系统的唯一启动点。
    "app.api.server:app" 表示寻找 app/api/server.py 文件中的 app 对象。
    reload=True 意味着你在开发时修改任何 Python 代码，服务器都会热更新，无需手动重启。
    """
    print("🚀 正在启动 Nexus 智能体集群...")
    uvicorn.run("app.api.server:app", host="0.0.0.0", port=8000, reload=True)