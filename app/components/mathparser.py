import math
import ast
import operator as op

class MathParser:
    """Basic math expression parser with variable support.

    Courtesy of `user3240484 <https://stackoverflow.com/a/69540962>`_.

    :param vars: Mapping where ``vars[name] -> numeric value`` used for evaluation.
    :param math: If ``True`` (default), expose functions/constants from ``math`` module.

    Example
    -------
    >>> data = {'r': 3.4, 'theta': 3.141592653589793}
    >>> parser = MathParser(data)
    >>> round(parser.parse('r*cos(theta)'), 1)
    -3.4
    >>> data['theta'] = 0.0
    >>> parser.parse('r*cos(theta)')
    3.4
    """
        
    _operators2method = {
        ast.Add: op.add, 
        ast.Sub: op.sub, 
        ast.BitXor: op.xor, 
        ast.Or:  op.or_, 
        ast.And: op.and_, 
        ast.Mod:  op.mod,
        ast.Mult: op.mul,
        ast.Div:  op.truediv,
        ast.Pow:  op.pow,
        ast.FloorDiv: op.floordiv,              
        ast.USub: op.neg, 
        ast.UAdd: lambda a:a  
    }
    
    def __init__(self, vars, math=True):
        self._vars = vars
        if not math:
            self._alt_name = self._no_alt_name
        
    def _Name(self, name):
        """Look up a variable name in the parser's namespace.

        :param name: Variable name to look up.
        :returns: Value from ``vars`` mapping or math module.
        :raises NameError: If name is not found.
        """
        try:
            return  self._vars[name]
        except KeyError:
            return self._alt_name(name)
                
    @staticmethod
    def _alt_name(name):
        """Look up a name in the math module if not found in ``vars``.

        :param name: Name to look up in math module.
        :returns: Math module function/constant.
        :raises NameError: If name starts with underscore or isn't in math.
        """
        if name.startswith("_"):
            raise NameError(f"{name!r}") 
        try:
            return  getattr(math, name)
        except AttributeError:
            raise NameError(f"{name!r}") 
    
    @staticmethod
    def _no_alt_name(name):
        raise NameError(f"{name!r}") 
    
    def eval_(self, node):
        """Evaluate an AST node recursively.

        :param node: AST node to evaluate.
        :returns: Result of evaluating the expression.
        :raises TypeError: If node type is not supported.
        """
        if isinstance(node, ast.Expression):
            return self.eval_(node.body)
        if isinstance(node, ast.Num): # <number>
            return node.n
        if isinstance(node, ast.Name):
            return self._Name(node.id) 
        if isinstance(node, ast.BinOp):            
            method = self._operators2method[type(node.op)]                      
            return method( self.eval_(node.left), self.eval_(node.right) )            
        if isinstance(node, ast.UnaryOp):             
            method = self._operators2method[type(node.op)]  
            return method( self.eval_(node.operand) )
        if isinstance(node, ast.Attribute):
            return getattr(self.eval_(node.value), node.attr)
            
        if isinstance(node, ast.Call):            
            return self.eval_(node.func)( 
                      *(self.eval_(a) for a in node.args),
                      **{k.arg:self.eval_(k.value) for k in node.keywords}
                     )           
            return self.Call( self.eval_(node.func), tuple(self.eval_(a) for a in node.args))
        else:
            raise TypeError(node)
    
    def parse(self, expr):
        """Parse and evaluate a mathematical expression string.

        :param expr: Expression string to parse.
        :returns: Numerical result of evaluating the expression.
        """
        return  self.eval_(ast.parse(expr, mode='eval'))          
    