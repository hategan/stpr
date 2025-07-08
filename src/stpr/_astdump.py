import ast
import sys
from io import StringIO
from typing import List, TextIO


TAB_SZ = 2


def _fmtconst(value: object) -> str:
    if isinstance(value, str):
        return '\'' + value + '\''
    else:
        return str(value)


def _get_func(node: ast.AST) -> str:
    if node.__class__ == ast.Name:
        return node.id
    if node.__class__ == ast.Attribute:
        return _get_func(node.value) + '.' + node.attr
    if node.__class__ == ast.Constant:
        return node.value
    if node.__class__ == ast.Subscript:
        return f'{_get_func(node.value)}[{_get_func(node.slice)}]'
    raise Exception('?? %s' % node)


def _astdumpc(nodes: List[ast.AST], indent, stream):
    if nodes is None:
        return
    for node in nodes:
        _astdumpi(node, indent, stream)


def _header(label: str, indent: int, stream):
    stream.write(' ' * indent * TAB_SZ)
    stream.write('%s\n' % label)


def _line_info(node: ast.AST) -> str:
    return '  \033[37mline(s): %s-%s, col(s): %s-%s\033[0m' % (getattr(node, 'lineno', '?'),
                                                getattr(node, 'end_lineno', '?'),
                                                getattr(node, 'col_offset', '?'),
                                                getattr(node, 'end_col_offset', '?'))


def _write_indent(indent: int, stream: TextIO) -> None:
    stream.write(' ' * indent * TAB_SZ)


def _astdumpi_generic(node: ast.AST, indent: int, stream: TextIO) -> None:
    _write_indent(indent, stream)
    cls = node.__class__
    stream.write(f'{node.__class__.__name__}\t{_line_info(node)}\n')
    for field in node._fields:
        val = getattr(node, field, None)
        if isinstance(val, list):
            _astdumpc(val, indent + 1, stream)
        elif isinstance(val, ast.AST):
            _astdumpi(val, indent + 1, stream)
        elif val is None:
            pass
        else:
            stream.write(' ' * (indent + 1) * TAB_SZ)
            stream.write(f'y {str(node)}\n')


_GENERICS = {ast.Expr, ast.Await, ast.With, ast.For, ast.AsyncFor, ast.AsyncWith, ast.Subscript,
             ast.Assign, ast.Lambda, ast.Return, ast.Nonlocal, ast.Raise, ast.If, ast.While,
             ast.withitem, ast.Module, ast.Expression}


def _astdumpi(node: ast.AST, indent, stream):
    cls = node.__class__
    if cls in _GENERICS:
        _astdumpi_generic(node, indent, stream)
        return

    _write_indent(indent, stream)
    if cls == ast.FunctionDef:
        stream.write('FunctionDef[%s] %s\n' % (node.name, _line_info(node)))
        _astdumpc(node.body, indent + 1, stream)
    elif cls == ast.AsyncFunctionDef:
        stream.write('AsyncFunctionDef[%s] %s\n' % (node.name, _line_info(node)))
        _astdumpc(node.body, indent + 1, stream)
    elif cls == ast.Call:
        stream.write('Call[%s: %s] %s\n' % (_get_func(node.func), getattr(node, '_type', None),
                                            _line_info(node)))
        if node.args:
            _write_indent(indent + 1, stream)
            stream.write('args:\n')
            _astdumpc(node.args, indent + 2, stream)
        if node.keywords:
            _write_indent(indent + 1, stream)
            stream.write('kwargs:\n')
            _astdumpc(node.keywords, indent + 2, stream)
    elif cls == ast.Constant:
        stream.write('Constant[%s]\n' % _fmtconst(node.value))
    elif cls == ast.Tuple:
        stream.write('Tuple[ctx=%s]\n' % getattr(node, 'ctx', '<!MissingCtx>'))
        _astdumpc(node.elts, indent + 1, stream)
    elif cls == ast.Attribute:
        stream.write('Attribute[%s] %s\n' % (node.attr, _line_info(node)))
        _astdumpi(node.value, indent + 1, stream)
    elif cls == ast.Name:
        stream.write('Name[%s: %s, ctx=%s] %s\n' % (getattr(node, 'id', '<MissingID>'),
                                                    getattr(node, '_type', None),
                                                    getattr(node, 'ctx', '<!MissingCtx>'),
                                                    _line_info(node)))
    else:
        stream.write(str(node))
        stream.write('\n')


def astdump(root: ast.AST, stream=sys.stdout) -> None:
    """
    Prints an AST.

    :param root: The root node of the AST.
    :param stream: An optional stream to print to. If not specified, print to stdout.
    """
    _astdumpi(root, 0, stream)


def astdumps(root: ast.AST) -> str:
    """
    Returns a string representation of an AST.

    This function uses :func:`astdump`.

    :param root: The root node of the AST.
    :return: A string representing the AST.
    """
    f = StringIO()
    astdump(root, f)
    return f.getvalue()
