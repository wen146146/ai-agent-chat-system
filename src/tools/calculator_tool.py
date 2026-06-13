import math
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.tools import tool


class CalculatorInput(BaseModel):
    """
    计算器工具输入参数模型。
    LLM 会根据此 Schema 生成结构化的函数调用参数。
    """
    operation: Literal[
        "add", "subtract", "multiply", "divide",
        "power", "sqrt", "log", "factorial", "sin", "cos"
    ] = Field(
        description="运算类型: add(加) subtract(减) multiply(乘) divide(除) "
                    "power(幂) sqrt(开方) log(对数) factorial(阶乘) sin(正弦) cos(余弦)"
    )
    a: float = Field(description="第一个操作数（必填）")
    b: float = Field(default=0, description="第二个操作数（sqrt/factorial/sin/cos 不需要）")


@tool(args_schema=CalculatorInput)
def calculator(operation: str, a: float, b: float = 0) -> str:
    """
    执行数学计算，支持 add/subtract/multiply/divide/power/sqrt/log/factorial/sin/cos。
    输入 operation 选择运算类型, a 为第一操作数, b 为第二操作数（部分运算不需要 b）。
    """
    op = operation.lower()
    try:
        if op == "add":
            result = a + b
            formula = f"{a} + {b} = {result}"
        elif op == "subtract":
            result = a - b
            formula = f"{a} - {b} = {result}"
        elif op == "multiply":
            result = a * b
            formula = f"{a} × {b} = {result}"
        elif op == "divide":
            if b == 0:
                return "[错误] 除数不能为 0"
            result = a / b
            formula = f"{a} ÷ {b} = {result}"
        elif op == "power":
            result = math.pow(a, b)
            formula = f"{a} ^ {b} = {result}"
        elif op == "sqrt":
            if a < 0:
                return "[错误] 不能对负数开平方"
            result = math.sqrt(a)
            formula = f"√{a} = {result}"
        elif op == "log":
            if a <= 0 or b <= 1:
                return "[错误] 对数运算要求真数>0 且底数>1"
            result = math.log(a, b)
            formula = f"log_{b}({a}) = {result}"
        elif op == "factorial":
            if a < 0 or a != int(a):
                return "[错误] 阶乘只支持非负整数"
            result = math.factorial(int(a))
            formula = f"{int(a)}! = {result}"
        elif op == "sin":
            result = math.sin(math.radians(a))
            formula = f"sin({a}°) = {result:.6f}"
        elif op == "cos":
            result = math.cos(math.radians(a))
            formula = f"cos({a}°) = {result:.6f}"
        else:
            return f"[错误] 不支持的运算类型: {op}"

        rounded = round(result, 8)
        return f"{formula}\n结果: {rounded}"
    except Exception as e:
        return f"[错误] 计算异常: {str(e)}"
