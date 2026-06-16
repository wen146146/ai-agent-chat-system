import math
from typing import Union
from pydantic import BaseModel, Field
from langchain_core.tools import tool


class CalculatorInput(BaseModel):
    """数学计算工具输入参数模型"""
    operation: str = Field(description="运算类型：add(加)、sub(减)、mul(乘)、div(除)、pow(幂)、sqrt(开方)、log(对数)、factorial(阶乘)、sin/cos/tan(三角函数)")
    a: Union[int, float] = Field(description="第一个数字或三角函数的弧度值")
    b: Union[int, float] = Field(default=0, description="第二个数字（指数/对数的底数，默认为0）")


@tool(args_schema=CalculatorInput)
def calculator(operation: str, a: Union[int, float], b: Union[int, float] = 0) -> str:
    """
    数学计算工具，支持加、减、乘、除、幂、开方、对数、阶乘、三角函数等运算。
    适用于需要数学计算的场景。
    """
    try:
        if operation == "add":
            result = a + b
            return f"{a} + {b} = {result}"
        elif operation == "sub":
            result = a - b
            return f"{a} - {b} = {result}"
        elif operation == "mul":
            result = a * b
            return f"{a} × {b} = {result}"
        elif operation == "div":
            if b == 0:
                return "[错误] 除数不能为 0"
            result = a / b
            return f"{a} ÷ {b} = {result}"
        elif operation == "pow":
            result = a ** b
            return f"{a} ^ {b} = {result}"
        elif operation == "sqrt":
            if a < 0:
                return "[错误] 不能对负数开方"
            result = math.sqrt(a)
            return f"√{a} = {result}"
        elif operation == "log":
            if a <= 0 or b <= 0 or b == 1:
                return "[错误] 对数参数必须为正数且底数不能为1"
            result = math.log(a, b)
            return f"log_{b}({a}) = {result}"
        elif operation == "factorial":
            if a < 0 or not float(a).is_integer():
                return "[错误] 阶乘只能用于非负整数"
            result = math.factorial(int(a))
            return f"{a}! = {result}"
        elif operation == "sin":
            result = math.sin(a)
            return f"sin({a}) = {result}"
        elif operation == "cos":
            result = math.cos(a)
            return f"cos({a}) = {result}"
        elif operation == "tan":
            result = math.tan(a)
            return f"tan({a}) = {result}"
        else:
            return f"[错误] 不支持的运算: {operation}，支持的运算: add/sub/mul/div/pow/sqrt/log/factorial/sin/cos/tan"
    except Exception as e:
        return f"[错误] 计算失败: {str(e)}"
