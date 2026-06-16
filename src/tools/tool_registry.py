from typing import Dict, List, Optional, Any, Callable
from langchain_core.tools import BaseTool
from pydantic import BaseModel


class DynamicToolInput(BaseModel):
    """动态参数模型，根据工具自身的 args_schema 自适应校验"""
    pass


class ToolRegistry:
    """
    工具注册中心：管理所有工具的注册、查询、调用。
    兼容 LangChain @tool 装饰器创建的 StructuredTool 实例。
    单例模式，全局唯一实例。
    """

    _instance: Optional["ToolRegistry"] = None
    _tools: Dict[str, BaseTool]

    def __new__(cls) -> "ToolRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools = {}
        return cls._instance

    def register(self, tool: BaseTool) -> None:
        """注册一个工具实例（@tool 装饰后的函数或 BaseTool 子类实例）"""
        if not isinstance(tool, BaseTool):
            raise TypeError(f"工具必须是 BaseTool 的实例, 收到: {type(tool)}")
        if tool.name in self._tools:
            raise ValueError(f"工具 [{tool.name}] 已经注册过了")
        self._tools[tool.name] = tool
        print(f"[注册] 工具已加载: {tool.name}")

    def unregister(self, name: str) -> None:
        """移除一个已注册的工具"""
        if name in self._tools:
            del self._tools[name]
            print(f"[移除] 工具已卸载: {name}")

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """按名称获取工具实例"""
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """导出所有工具的 JSON Schema 列表（供 LLM function calling 使用）"""
        result = []
        for tool in self._tools.values():
            result.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.args_schema.model_json_schema()
            })
        return result

    def list_tool_names(self) -> List[str]:
        """列出所有已注册工具的名称"""
        return list(self._tools.keys())

    def invoke(self, name: str, **kwargs: Any) -> str:
        """
        调用指定工具，返回执行结果字符串。
        如果工具不存在或执行出错，返回错误描述。
        """
        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(self._tools.keys()) if self._tools else "无"
            return f"[错误] 工具 [{name}] 不存在, 当前可用工具: {available}"
        try:
            return tool.invoke(kwargs)
        except Exception as e:
            return f"[错误] 工具 [{name}] 执行失败: {str(e)}"

    def collect_native_tools(self) -> List[BaseTool]:
        """
        收集所有原生 LangChain @tool 实例。
        供 model.bind_tools() / create_react_agent() 直接使用。
        LLM 会自动从每个工具的 args_schema 获取参数结构。
        """
        from src.tools.calculator_tool import calculator
        from src.tools.web_search_tool import web_search, web_fetch
        from src.tools.rag_retriever_tool import rag_retrieve
        from src.tools.memory.episodic_memory_tool import (
            episodic_memory_save,
            episodic_memory_search,
            episodic_memory_delete,
        )
        from src.tools.memory.semantic_memory_tool import (
            semantic_memory_save,
            semantic_memory_search,
            semantic_memory_delete,
            semantic_memory_count,
        )
        from src.tools.file_tools import read_file, write_file, search_files, list_directory
        from src.tools.shell_tools import run_command
        from src.tools.app_tools import open_application, list_applications
        from src.tools.system_tools import get_system_info, get_process_list

        return [
            calculator,
            web_search,
            web_fetch,
            rag_retrieve,
            episodic_memory_save,
            episodic_memory_search,
            episodic_memory_delete,
            semantic_memory_save,
            semantic_memory_search,
            semantic_memory_delete,
            semantic_memory_count,
            # Phase 2 新增工具
            read_file, write_file, search_files, list_directory,
            run_command,
            open_application, list_applications,
            get_system_info, get_process_list,
        ]

    @classmethod
    def get_instance(cls) -> "ToolRegistry":
        """获取单例实例（比 ToolRegistry() 更语义化）"""
        return cls()
