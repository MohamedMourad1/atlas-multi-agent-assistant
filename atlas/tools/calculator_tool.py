"""
Sandboxed Calculator Tool for Atlas.
Allows agents to evaluate arithmetic, unit conversions, and percentages safely without arbitrary code execution.
"""

import ast
import operator
from typing import Union, Dict, Any

# Supported safe operators
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


class SandboxedCalculator:
    """Safely evaluates mathematical expressions using Python AST."""

    def evaluate(self, expression: str) -> Dict[str, Any]:
        """Safely compute the mathematical result of an expression."""
        try:
            cleaned = expression.strip().replace("^", "**").replace("×", "*").replace("÷", "/")
            node = ast.parse(cleaned, mode="eval")
            result = self._eval_node(node.body)
            return {
                "expression": expression,
                "result": round(float(result), 4) if isinstance(result, (int, float)) else str(result),
                "status": "success",
            }
        except Exception as e:
            return {
                "expression": expression,
                "result": None,
                "error": f"Evaluation error: {str(e)}",
                "status": "failed",
            }

    def _eval_node(self, node: ast.AST) -> Union[int, float]:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant type: {type(node.value)}")
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type in SAFE_OPERATORS:
                left = self._eval_node(node.left)
                right = self._eval_node(node.right)
                return SAFE_OPERATORS[op_type](left, right)
            raise ValueError(f"Unsupported binary operator: {op_type}")
        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type in SAFE_OPERATORS:
                operand = self._eval_node(node.operand)
                return SAFE_OPERATORS[op_type](operand)
            raise ValueError(f"Unsupported unary operator: {op_type}")
        else:
            raise ValueError(f"Unsupported AST node: {type(node)}")


# Global singleton calculator
safe_calculator = SandboxedCalculator()
