import ast
import asyncio
import inspect
import textwrap
import types
from asyncio import Task
from contextlib import AbstractAsyncContextManager
from enum import Enum
from typing import Tuple, List, Optional, Coroutine, Type, Dict, Callable, Set, Union, Iterable

from stpr._astdump import astdump, astdumps
from stpr._debug import debug_print, Color, _ts, DEBUG, _print

_SAFE_MODULES = set()
_SAFE_MODULES.add('stpr')
_SAFE_MODULES.add('stpr.transformer')
_SAFE_MODULES.add('stpr.types')
_SAFE_MODULES.add('stpr.reactive')


class _Ref:
    def __init__(self, value: Optional[object] = None, _type: Optional[Type] = None) -> None:
        self.value = value
        if type is None and value is not None:
            self.type = type(value)
        else:
            self.type = _type

    def is_coro(self) -> bool:
        return inspect.iscoroutinefunction(self.value)

    def is_async_ctx_manager(self) -> bool:
        if self.value:
            return hasattr(self.value, '__aenter__') and hasattr(self.value, '__aexit__')
        if self.type:
            return hasattr(self.type, '__aenter__') and hasattr(self.type, '__aexit__')

    def is_ctx_manager(self) -> bool:
        if self.value:
            return hasattr(self.value, '__enter__') and hasattr(self.value, '__exit__')
        if self.type:
            return hasattr(self.type, '__enter__') and hasattr(self.type, '__exit__')

    def __str__(self) -> str:
        return 'Ref[%s: %s]' % (self.value, self.type)


def _find_global(frame, name: str) -> object:
    try:
        return _Ref(frame.f_locals[name])
    except KeyError:
        try:
            return _Ref(frame.f_globals[name])
        except KeyError:
            try:
                return _Ref(frame.f_builtins[name])
            except KeyError:
                debug_print(f'No reference found for "{name}"', Color.CYAN)
                return None


def _get_type(node: ast.AST) -> Type:
    if hasattr(node, '_type'):
        return getattr(node, '_type')
    else:
        return None


def _transform_with(node: ast.With):
    return node

def _copy_params(src: ast.AST, dst: ast.AST) -> None:
    for field_name in src._fields:
        setattr(dst, field_name, getattr(src, field_name))
    for name in ['lineno', 'end_lineno', 'col_offset', 'end_col_offset']:
        if hasattr(src, name):
            val = getattr(src, name)
            setattr(dst, name, val)
    return dst


def _transform(node: ast.AST, frame, sp_mod_name: str):
    t = _Transformer(frame, sp_mod_name)
    t.visit(node)
    debug_print(frame, Color.BLUE)
    return node

def _is_async(node: ast.AST, outer_frame, locals: Dict[str, object]) -> bool:
    cls = node.__class__
    if cls == ast.Call:
        pass
    elif cls == ast.Constant:
        return True
    elif cls == ast.Attribute:
        pass
    elif cls == ast.Name:
        pass


_EXPR_ATTRS_LIST = {
    ast.BoolOp: ['values'],
    ast.Dict: ['keys', 'values'],
    ast.Set: ['elts'],
    ast.Compare: ['comparators'],
    ast.Call: ['args'],
    ast.JoinedStr: ['values'],
    ast.List: ['elts'],
    ast.Tuple: ['elts']
}

_EXPR_ATTRS = {
    ast.NamedExpr: ['target', 'value'],
    ast.BinOp: ['left', 'right'],
    ast.UnaryOp: ['operand'],
    ast.Lambda: ['body'],
    ast.IfExp: ['test', 'body', 'orelse'],
    ast.ListComp: ['elt'],
    ast.SetComp: ['elt'],
    ast.DictComp: ['key', 'value'],
    ast.Compare: ['left'],
    ast.Call: ['func'],
    ast.FormattedValue: ['value'],
    ast.Attribute: ['value'],
    ast.Subscript: ['value', 'slice'],
    ast.Starred: ['value']
}

_EXPR_OPT_ATTRS = {
    ast.FormattedValue: ['format_spec'],
    ast.Slice: ['lower', 'upper', 'step']
}

_EXPR_ATTRS_COMPS = {
    ast.ListComp: ['generators'],
    ast.SetComp: ['generators'],
    ast.DictComp: ['generators']
}

class ExprType:
    def __init__(self, has_heavy: Optional[bool] = False, has_async: Optional[bool] = False):
        self.has_heavy = has_heavy
        self.has_async = has_async
        self.is_coro = False

    def async_found(self) -> None:
        self.has_async = True

    def heavy_found(self) -> None:
        self.has_heavy = True

    def all_async(self) -> bool:
        return self.has_async and not self.has_heavy

    def all_heavy(self) -> bool:
        return self.has_heavy and not self.has_async

    def all_simple(self) -> bool:
        return not self.has_heavy and not self.has_async

    def combine(self, other: 'ExprType') -> None:
        self.has_heavy |= other.has_heavy
        self.has_async |= other.has_async

    def __iadd__(self, other: 'ExprType') -> None:
        self.combine(other)
        return self


def _analyze_all(l: List[ast.AST], tr: '_Transformer') -> ExprType:
    t = ExprType()
    for node in l:
        t += _analyze(node, tr)
    return t


def _analyze(expr: ast.AST, tr: '_Transformer') -> ExprType:
    t = ExprType()
    expr._a_type = t
    if expr is None:
        return t

    cls = expr.__class__

    if cls in _EXPR_ATTRS_LIST:
        for attr in _EXPR_ATTRS_LIST[cls]:
            sub_exprs = getattr(expr, attr)
            t += _analyze_all(sub_exprs, tr)

    if cls in _EXPR_ATTRS:
        for attr in _EXPR_ATTRS[cls]:
            sub_expr = getattr(expr, attr)
            t += _analyze(sub_expr, tr)

    if cls in _EXPR_OPT_ATTRS:
        for attr in _EXPR_OPT_ATTRS[cls]:
            sub_expr = getattr(expr, attr)
            if sub_expr is not None:
                t += _analyze(sub_expr, tr)

    if cls in _EXPR_ATTRS_COMPS:
        attr = _EXPR_ATTRS_COMPS[cls]
        (target, iter, ifs, comp_is_async) = getattr(expr, attr)
        if not comp_is_async:
            t += _analyze(iter, tr)
            t += _analyze_all(ifs, tr)

    if cls == ast.Await:
        # in principle we want to bring everything into a tree
        # of await coro, so if an expression is already that,
        # we don't need to do anything
        pass
    elif cls == ast.Call:
        ref = tr.get_ref(expr.func)
        if ref is None:
            t.heavy_found()
        elif ref.is_coro():
            t.async_found()
            t.is_coro = True
        else:
            t.heavy_found()

    return t


def _empty_ast_arguments():
    return ast.arguments(posonlyargs=[], args=[], defaults=[], kwonlyargs=[], kw_defaults=[])


def _make_args(args):
    return ast.arguments(posonlyargs=[], args=[ast.arg(arg=x) for x in args], defaults=[],
                         kwonlyargs=[], kw_defaults=[])


def _make_names(args):
    return [ast.Name(id=x, ctx=ast.Load()) for x in args]


def _clear_lineno(node: ast.AST) -> None:
    if hasattr(node, 'lineno'):
        delattr(node, 'lineno')
        delattr(node, 'end_lineno')
        delattr(node, 'col_offset')
        delattr(node, 'end_col_offset')
    for name in dir(node):
        val = getattr(node, name)
        if isinstance(val, ast.AST):
            _clear_lineno(val)


def _copy_lineinfo(src: ast.AST, dst: ast.AST) -> None:
    for attr in ['lineno', 'end_lineno', 'col_offset', 'end_col_offset']:
        setattr(dst, attr, getattr(src, attr, 0))


def _wrap_in_async_proc(nodes: List[ast.AST], tr: '_Transformer',
                        args: List[str] = []) -> List[ast.AST]:
    name = '__sp_proc_%s' % tr.next_tmp_index()
    def_ast = ast.AsyncFunctionDef(name=name, args=_make_args(args), body=nodes, decorator_list=[])
    invoke_ast = ast.Await(
        ast.Call(
            func=ast.Name(id=name, ctx=ast.Load()),
            args=_make_names(args),
            keywords=[]))
    return [def_ast, invoke_ast]


def _wrap_await(expr: ast.AST) -> ast.AST:
    return ast.Await(expr)


def _wrap_none(expr: ast.AST) -> ast.AST:
    return expr


def _wrap_par(expr: ast.AST) -> ast.AST:
    return ast.Call(func=ast.Attribute(value=ast.Name(id='__sp_pctx', ctx=ast.Load()),
                                       attr='_start', ctx=ast.Load()),
                    args=[expr], keywords=[])


def _wrap_in_lambda(expr: ast.AST, tr: '_Transformer',
                    wrapper: Callable[[ast.AST], ast.AST] = _wrap_await) -> ast.AST:
    return wrapper(
        ast.Call(
            func=ast.Attribute(value=ast.Name(id=tr.sp_mod_name, ctx=ast.Load()),
                               attr='_run_sync', ctx=ast.Load()),
            args=[ast.Lambda(args=_empty_ast_arguments(), body=expr)],
            keywords=[]))


def _sp_invoke(meth: str, node: ast.AST, tr: '_Transformer', args=[], kwargs=[]) -> ast.AST:
    copy = node.__class__()
    _copy_params(node, copy)
    return ast.Call(
        func=ast.Attribute(value=ast.Name(id=tr.sp_mod_name, ctx=ast.Load()),
                           attr=meth, ctx=ast.Load()),
        args=[copy] + args, keywords=kwargs)


def _ensure_async_expr_all(l: List[ast.AST], tr: '_Transformer') -> List[ast.AST]:
    for i in range(len(l)):
        l[i] = _ensure_async_expr(l[i], tr)

    return l


def _ensure_async_expr(expr: ast.AST, tr: '_Transformer', force: bool = False,
                       wrapper: Callable[[ast.AST], ast.AST] = _wrap_await) -> ast.AST:
    """
    Ensures that the expression is either awaitable or simple, so that it can be run outside the
    thread pool.
        c = a + b -> c = a + b
        c = H a + H b -> c = A C(H a + H b)
        c = A a + A b -> c = A a + a B
        c = H a + b -> c = A C(H a + b)
        c = A a + b -> c = A a + b
        c = A a + H b -> c = A a + A C(H b)

    :param node:
    :param tr:
    :return:
    """
    if expr is None:
        return None

    _analyze(expr, tr)

    return _ensure_async_expr_r(expr, tr, force=force, wrapper=wrapper)


def _ensure_async_expr_r(expr: ast.AST, tr: '_Transformer', force: bool = False,
                         wrapper: Callable[[ast.AST], ast.AST] = _wrap_await) -> ast.AST:
    # we know we have both async branches and heavy branches
    # we change heavy branches H x to await C(H x) at the thickest point while
    # adding await to all coro invocations

    if expr._a_type.all_simple():
        if force:
            return _wrap_in_lambda(expr, tr, wrapper)
        else:
            return expr
    if expr._a_type.all_heavy():
        return _wrap_in_lambda(expr, tr, wrapper)

    cls = expr.__class__

    if cls in _EXPR_ATTRS_LIST:
        for attr in _EXPR_ATTRS_LIST[cls]:
            sub_exprs = getattr(expr, attr)
            new_nodes = _ensure_async_expr_all(sub_exprs, tr)
            setattr(expr, attr, new_nodes)

    if cls in _EXPR_ATTRS:
        for attr in _EXPR_ATTRS[cls]:
            sub_expr = getattr(expr, attr)
            new_nodes = _ensure_async_expr(sub_expr, tr)
            setattr(expr, attr, new_nodes)

    if cls in _EXPR_ATTRS_COMPS:
        attr = _EXPR_ATTRS_COMPS[cls]
        (target, iter, ifs, comp_is_async) = getattr(expr, attr)
        if not comp_is_async:
            iter = _ensure_async_expr(iter, tr)
            ifs = _ensure_async_expr_all(ifs, tr)

    return expr


def _make_call(obj_name: str, method_name: str, args: Optional[ast.AST] = None) -> ast.AST:
    return ast.Call(
        func=ast.Attribute(
            value=ast.Name(id=obj_name, ctx=ast.Load()),
            attr=method_name,
            ctx=ast.Load()),
        args=[] if args is None else [args], keywords=[])


def _assign(name: str, value: ast.AST) -> ast.AST:
    return ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())],
                      value=value)


class _ContextType(Enum):
    SEQ = 0
    PAR = 1
    DEF = 2


class _Local:
    def __init__(self, name: str, value: Optional[object] = None, type: Optional[type] = None,
                 awaited: bool = False, is_param: bool = False, used: bool = False):
        self.name = name
        self.value = value
        self.type = type
        self.awaited = awaited
        self.is_param = is_param
        self.used = used

class _Locals:
    def __init__(self):
        self._vars = {}

    def add_var(self, name: str, value: Optional[object] = None, type: Optional[Type] = None,
                awaited: bool = False, is_param: bool = False, used: bool = False) -> None:
        self.vars[name] = _Local(name, value, type, awaited, is_param, used)
        debug_print('Adding var %s: %s = %s, awaited: %s' % (name, type, value, awaited), Color.BLUE)

    def is_awaited(self, name: str) -> bool:
        return name in self._vars and self._vars[name].awaited

    def is_used(self, name: str) -> bool:
        return name in self._vars and self._vars[name].used

    def is_param(self, name: str) -> bool:
        return name in self._vars and self._vars[name].is_param

    def set_awaited(self, name: str) -> None:
        self._vars[name].awaited = True

    def set_used(self, name: str) -> None:
        self._vars[name].used = True

    def get_type(self, name: str) -> Type:
        try:
            return self._vars[name].type
        except KeyError:
            return None

    @property
    def vars(self) -> Dict[str, object]:
        return self._vars


class _Context():
    def __init__(self, parent: '_Context', type: _ContextType) -> None:
        self.parent = parent
        self.type = type
        if type == _ContextType.PAR:
            self.locals = _Locals()
        elif parent is not None and type != _ContextType.DEF:
            self.locals = parent.locals
        else:
            self.locals = _Locals()
        self.nodes = []
        self.def_index = 0

    def add_local(self, name: str, value: Optional[object] = None, type: Optional[Type] = None,
                  awaited: bool = False, is_param: bool = False):
        self.locals.add_var(name, value, type, awaited, is_param)

    def is_awaited(self, node: ast.Name) -> bool:
        return self.locals.is_awaited(node.id)

    def is_used(self, node: ast.Name) -> bool:
        return self.locals.is_used(node.id)

    def is_param(self, node: ast.Name) -> bool:
        return self.locals.is_param(node.id)

    def set_awaited(self, node: ast.Name) -> None:
        self.locals.set_awaited(node.id)

    def set_used(self, node: ast.Name) -> None:
        self.locals.set_used(node.id)

    def get_type(self, node: ast.Name) -> Type:
        return self.locals.get_type(node.id)

    def get_local(self, name: str) -> _Ref:
        if name in self.locals.vars:
            return _Ref(self.locals.vars[name], self.locals.get_type(name))
        else:
            return None

    def var_names(self) -> Set[str]:
        return self.locals.vars.keys()


class _Transformer(ast.NodeTransformer):

    def __init__(self, outer_frame, sp_mod_name: str):
        self.outer_frame = outer_frame
        self.sp_mod_name = sp_mod_name
        self.context_stack = []
        self.crt_context = None

    def get_global(self, name: str) -> _Ref:
        return _find_global(self.outer_frame, name)

    def _find_local(self, ctx: _Context, name: str) -> _Ref:
        ref = ctx.get_local(name)
        if ref:
            return ref
        if ctx.parent:
            return self._find_local(ctx.parent, name)
        else:
            return None

    def get_local(self, name: str) -> _Ref:
        return self._find_local(self.crt_context, name)

    def _get_member(self, cls, name):
        members = inspect.getmembers(cls)
        for n, v in members:
            if name == n:
                return v
        return None

    def get_ref(self, node: ast.AST) -> _Ref:
        if isinstance(node, ast.Name):
            ref = self.get_local(node.id)
            if ref:
                return ref
            else:
                return self.get_global(node.id)
        elif isinstance(node, ast.Attribute):
            ref = self.get_ref(node.value)
            if ref:
                if ref.value:
                    try:
                        return _Ref(getattr(ref.value, node.attr))
                    except AttributeError:
                        pass
                elif ref.type:
                    if inspect.isclass(ref.type):
                        return _Ref(self._get_member(ref.type, node.attr))
                    else:
                        return None
                else:
                    return None
            else:
                return None
        else:
            return None

    def _enter_context(self, type: _ContextType = _ContextType.SEQ) -> None:
        self.crt_context = _Context(self.crt_context, type)
        self.context_stack.append(self.crt_context)

    def _exit_context(self) -> _Context:
        r = self.context_stack.pop()
        if len(self.context_stack) > 0:
            self.crt_context = self.context_stack[-1]
        else:
            self.crt_context = None
        return r

    def _seq(self, node: ast.AST, expr: Optional[ast.AST], var: Optional[ast.AST]) -> ast.AST:
        self._enter_context(_ContextType.SEQ)
        node.body = self._visit_nodes(node.body, self.crt_context.nodes)
        self._exit_context()
        return node

    def _parallel(self, node: ast.AST, expr: Optional[ast.AST], var: Optional[ast.AST]) -> ast.AST:
        new_node = ast.AsyncWith()
        _copy_params(node, new_node)
        node = new_node
        wi = node.items[0]
        if wi.optional_vars is not None:
            raise Exception('No variable allowed in parallel with statement')
        wi.optional_vars = ast.Name(id='__sp_pctx', ctx=ast.Store())
        node.body = self._make_parallel(node.body)
        return node

    def _fork(self, node: ast.AST, expr: Optional[ast.AST], var: Optional[ast.AST]) -> ast.AST:
        wi = node.items[0]
        if wi.optional_vars is not None:
            raise Exception('No variable allowed in fork with statement')
        node.body = self._make_fork(node.body)
        return node

    def _parallelFor(self, node: ast.AST, expr: Optional[ast.AST], var: Optional[ast.AST]) -> ast.AST:
        if len(expr.args) != 1:
            raise Exception('Missing range argument to parallelFor')
        arg = expr.args[0]
        if var is None:
            var = ast.Name(id='_', ctx=ast.Store())
        self.crt_context.add_local(var.id, awaited=True)
        iter = _make_call(self.sp_mod_name, '_to_aiter', arg)
        afor = ast.AsyncFor(target=var, iter=iter, body=node.body, orelse=[])

        r = ast.AsyncWith()
        wi = ast.withitem()
        wi.context_expr = _make_call(self.sp_mod_name, 'parallel')
        wi.optional_vars = ast.Name(id='__sp_pctx', ctx=ast.Store())
        r.items = [wi]
        r.body = [afor]
        self._enter_context(_ContextType.PAR)
        afor.body = self._visit_nodes(afor.body)
        exports = self.crt_context.var_names()
        if len(exports) > 0:
            afor.body.insert(0, ast.Nonlocal(list(exports)))
            for exp in exports:
                if exp not in self.crt_context.parent.var_names():
                    r.body.insert(0, _assign(exp, ast.Constant(value=None)))

        afor.body = _wrap_in_async_proc(afor.body, self, args=[var.id])
        afor.body[1] = ast.Expr(_wrap_par(afor.body[1].value))
        return r

    @property
    def crt_nodes(self):
        return self.crt_context.nodes

    def next_tmp_index(self):
        self.crt_context.def_index += 1
        return self.crt_context.def_index

    def _new_with(self, id: str, body: List[ast.AST]) -> ast.AST:
        with_ast = ast.With(items=[
            ast.withitem(
                ast.Attribute(
                    value=ast.Name(id=self.sp_mod_name, ctx=ast.Load()),
                    attr=id,
                    ctx=ast.Load()
                )
            )
        ], body=body)
        with_ast.lineno = 0
        with_ast.end_lineno = 0
        return with_ast

    def _new_seq(self, body: List[ast.AST]) -> ast.AST:
        return self._new_with('seq', body)

    def _has_sp_decorator(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
        return True
        for decorator in node.decorator_list:
            ref = self.get_ref(decorator)
            if ref is not None and ref.value == fn:
                return True
        return False

    def visit_FunctionDef(self, node):
        self._enter_context(_ContextType.SEQ)
        self.crt_context.add_local(node.name, type=Coroutine, awaited=True)
        try:
            assert isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef)
            if self._has_sp_decorator(node):
                node = _copy_params(node, ast.AsyncFunctionDef())

                if len(node.body) == 0:
                    raise SyntaxError()

                for arg in node.args.posonlyargs + node.args.args:
                    self.crt_context.add_local(arg.arg, None, None, False, True)

                node.body = self._visit_nodes(node.body)
                node.body = self.crt_nodes + node.body

            return node
        finally:
            self._exit_context()

    def _await(self, name: str, src: ast.AST) -> ast.AST:
        ref = ast.Name(id=name, ctx = ast.Load())
        await_what = _sp_invoke('_await', ref, self)
        aw = ast.Await(await_what)
        _copy_lineinfo(src, aw)
        return ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=aw)

    def visit_Name(self, node):
        self.generic_visit(node)

        if not isinstance(node.ctx, ast.Load):
            return node
        type = self.crt_context.get_type(node)
        if type is not None:
            node._type = type
        if self.crt_context.is_awaited(node):
            return node

        debug_print('Type for %s is %s' % (node.id, type), Color.BLUE)

        ref = self.get_ref(node)

        if ref is None:
            self.crt_nodes.append(self._await(node.id, node))
            self.crt_context.set_awaited(node)
        elif self.crt_context.is_param(node) and not self.crt_context.is_awaited(node):
            self.crt_nodes.insert(0, self._await(node.id, node))
            self.crt_context.set_awaited(node)
        else:
            pass

        return node


    def _is_safe(self, mod: Optional[types.ModuleType]) -> bool:
        if mod is None:
            return False
        else:
            return mod.__name__ in _SAFE_MODULES

    def visit_Call(self, node: ast.AST) -> ast.AST:
        # heavy(heavy) ->
        #   def __sp_fn_n():
        #       heavy(heavy)
        #   await sp._call(__sp_fn_n)
        #
        # heavy(simple) ->
        #   await sp._call(heavy, simple)
        #
        # coro(heavy) ->
        #   __sp_tmp_n = sp._call(heavy)
        #   await coro(__sp_tmp_n)

        debug_print('Call %s' % astdumps(node.func), Color.BLUE)
        ref = self.get_ref(node.func)

        if ref is None:
            self.generic_visit(node)
            return ast.Await(_sp_invoke('_call', node.func, self, args=node.args, kwargs=node.keywords))
        elif ref.value in _SP_FNS:
            if ref.value == parallel:
                return self._make_parallel_fn(node)
            elif ref.value == race:
                return self._make_race_fn(node)
            else:
                raise Exception(f'Unhandled sp function: {ref.value}')
        elif ref.is_coro():
            return ast.Await(node)
        elif inspect.isclass(ref.value) and self._is_safe(inspect.getmodule(ref.value)):
            self.generic_visit(node)
            node._type = ref.value
            return node
        elif inspect.isfunction(ref.value) and self._is_safe(inspect.getmodule(ref.value)):
            self.generic_visit(node)
            node._type = ref.value
            return node
        elif inspect.ismethod(ref.value) and self._is_safe(inspect.getmodule(ref.value)):
            self.generic_visit(node)
            node._type = ref.value
            return node
        else:
            self.generic_visit(node)
            if hasattr(node.func, '_type'):
                type = node.func._type
                if type == Coroutine:
                    return ast.Await(node)
            return ast.Await(_sp_invoke('_call', node.func, self, args=node.args, kwargs=node.keywords))

    def _to_coro(self, node: ast.AST) -> ast.AST:
        if isinstance(node, ast.Call):
            ref = self.get_ref(node.func)
            if ref is None:
                self.generic_visit(node)
                return _sp_invoke('_call', node.func, self, args=node.args)
            elif ref.is_coro():
                return node

            self.generic_visit(node)
            if hasattr(node.func, '_type'):
                type = node.func._type
                if type == Coroutine:
                    return node
            return _sp_invoke('_call', node.func, self, args=node.args)
        else:
            tmp_name = '__sp_fn%s' % self.next_tmp_index()
            coro = ast.AsyncFunctionDef(name=tmp_name, args=_empty_ast_arguments(),
                                        decorator_list=[], returns=None,
                                        body=[ast.Return(value=node)])
            self.generic_visit(coro)

            self.crt_context.nodes.append(coro)
            return ast.Call(func=ast.Name(tmp_name, ctx=ast.Load()), args=[], keywords=[])

    def visit_With_item(self, node: ast.AST, item_ix: int) -> ast.AST:
        assert isinstance(node, ast.With)
        item = node.items[item_ix]
        if item.optional_vars:
            assert isinstance(item.optional_vars, ast.Name)
            assert self.crt_context is not None
            self.crt_context.add_local(item.optional_vars.id, awaited=True)
        if item_ix < len(node.items) - 1:
            body = [self.visit_With_item(node, item_ix + 1)]
            body_visited = True
        else:
            body = node.body
            body_visited = False
        expr = item.context_expr
        ref = None
        if isinstance(expr, ast.Attribute) or isinstance(expr, ast.Name):
            ref = self.get_ref(expr)
            if ref and ref.value in _SP_CMS:
                expr = ast.Call(func=expr)
                expr.args = []
                expr.keywords = []
                item.context_expr = expr
        if isinstance(expr, ast.Call):
            ref = self.get_ref(expr.func)
            val = ref.value if ref else None
            debug_print('Val: %s' % val, Color.BLUE)
            if val in _SP_CMS:
                if len(node.items) > 1:
                    raise Exception('%s cannot be used in a multi-item with' % val.__name__)
                tr = val._transformer()
                return tr(self, node, item.context_expr, item.optional_vars)
            elif (isinstance(val, type) and issubclass(val, AbstractAsyncContextManager)) or \
                inspect.iscoroutinefunction(val) or inspect.iscoroutine(val):
                r = ast.AsyncWith()
                _copy_params(node, r)
                r.items = [item]
                if body_visited:
                    r.body = body
                else:
                    r.body = self._visit_nodes(body)
                return r
            else:
                r = ast.AsyncWith()
                _copy_params(node, r)
                item.context_expr = _make_call(self.sp_mod_name, '_to_acm', item.context_expr)
                r.items = [item]
                if body_visited:
                    r.body = body
                else:
                    r.body = self._visit_nodes(body)
                return r
        else:
            if (ref is None) or (ref and (ref.is_async_ctx_manager() or not ref.is_ctx_manager())):
                r = ast.AsyncWith()
                _copy_params(node, r)
                if ref is None or (not ref.is_ctx_manager() and not ref.is_async_ctx_manager()):
                    # unknown; coerce to async context manager
                    item.context_expr = _sp_invoke('_to_acm', item.context_expr, self)
                r.items = [item]
                if body_visited:
                    r.body = body
                else:
                    r.body = self._visit_nodes(body)
                return r
            else:
                self.generic_visit(node)
                return node

    def visit_With(self, node: ast.AST):
        debug_print('Visiting with node %s' % node, Color.BLUE)
        return self.visit_With_item(node, 0)

    def visit_For(self, node):
        r = ast.AsyncFor()
        _copy_params(node, r)
        names = None
        if isinstance(r.target, ast.Tuple):
            names = r.target.elts
        elif isinstance(r.target, ast.Name):
            names = [r.target]
        for name in names:
            assert isinstance(name, ast.Name)
            self.crt_context.add_local(name.id, awaited=True)
        self.generic_visit(r)
        r.iter = _make_call(self.sp_mod_name, '_to_aiter', r.iter)
        return r

    def visit_Assign(self, node):
        node.value = self.visit(node.value)
        if len(node.targets) == 1:
            if isinstance(node.targets[0], ast.Name):
                var = node.targets[0].id
                self.crt_context.add_local(var, type=_get_type(node.value), awaited=True)
            elif isinstance(node.targets[0], ast.Tuple):
                for elt in node.targets[0].elts:
                    if isinstance(elt, ast.Name):
                        var = elt.id
                        self.crt_context.add_local(var, type=_get_type(node.value), awaited=True)
        return node

    def visit_AnnAssign(self, node):
        return self.visit_AugAssign(node)

    def visit_AugAssign(self, node):
        node.value = self.visit(node.value)
        if isinstance(node.target, ast.Name):
            var = node.target.id
            self.crt_context.add_local(var, type=_get_type(node.value), awaited=True)
        return node

    def visit_If(self, node):
        self.generic_visit(node.test)
        self._enter_context()
        node.body = self._visit_nodes(node.body, self.crt_nodes)
        self._exit_context()
        self._enter_context()
        node.orelse = self._visit_nodes(node.orelse, self.crt_nodes)
        self._exit_context()
        return node

    def visit_While(self, node):
        return self.visit_If(node)

    def visit_Try(self, node):
        for handler in node.handlers:
            if handler.name is not None:
                self.crt_context.add_local(handler.name, awaited=True)
        return self.generic_visit(node)

    def _make_parallel_fn(self, call: ast.Call) -> ast.AST:
        if len(call.keywords) > 0:
            raise Exception('parallel() does not support keyword arguments.')
        if len(call.args) < 2:
            raise Exception('parallel() needs at least two arguments.')
        call.func = ast.Attribute(value=call.func, attr='_fn', ctx=ast.Load())
        for i in range(len(call.args)):
            call.args[i] = self._to_coro(call.args[i])
        return ast.Await(call)

    def _make_race_fn(self, call: ast.Call) -> ast.AST:
        if len(call.keywords) > 0:
            raise Exception('race() does not support keyword arguments.')
        if len(call.args) < 2:
            raise Exception('race() needs at least two arguments.')
        call.func = ast.Attribute(value=call.func, attr='_fn', ctx=ast.Load())
        for i in range(len(call.args)):
            call.args[i] = self._to_coro(call.args[i])
        return ast.Await(call)

    def _make_parallel(self, nodes: List[ast.AST],
                       r_nodes: Optional[List[ast.AST]] = None) -> List[ast.AST]:
        if nodes is None:
            return None
        if r_nodes is None:
            r_nodes = []
        for node in nodes:
            tmp_name = '__sp_fn%s' % self.next_tmp_index()
            body = [node]
            self._enter_context(_ContextType.PAR)
            coro = ast.AsyncFunctionDef(name=tmp_name, args=_empty_ast_arguments(),
                                        decorator_list=[], returns = None,
                                        body=body)
            self.generic_visit(coro)
            exports = self.crt_context.var_names()
            if len(exports) > 0:
                body.insert(0, ast.Nonlocal(list(exports)))
                for exp in exports:
                    if exp not in self.crt_context.parent.var_names():
                        r_nodes.append(_assign(exp, ast.Constant(value=None)))
                        self.crt_context.parent.add_local(exp, awaited=True)
            r_nodes.append(coro)
            call = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='__sp_pctx', ctx=ast.Load()),
                    attr='_start',
                    ctx=ast.Load()),
                args=[ast.Call(func=ast.Name(id=tmp_name, ctx=ast.Load()), args=[], keywords=[])],
                keywords=[]
            )
            r_nodes.append(ast.Expr(call))
            self._exit_context()
        return r_nodes

    def _make_fork(self, nodes: List[ast.AST], r_nodes: Optional[List[ast.AST]] = None) -> List[ast.AST]:
        if nodes is None:
            return None
        if r_nodes is None:
            r_nodes = []
        for node in nodes:
            tmp_name = '__sp_fn%s' % self.next_tmp_index()
            body = [node]
            self._enter_context(_ContextType.PAR)
            coro = ast.AsyncFunctionDef(name=tmp_name, args=_empty_ast_arguments(),
                                        decorator_list=[], returns = None,
                                        body=body)
            self.generic_visit(coro)
            exports = self.crt_context.var_names()
            if len(exports) > 0:
                body.insert(0, ast.Nonlocal(list(exports)))
                for exp in exports:
                    if exp not in self.crt_context.parent.var_names():
                        r_nodes.append(_assign(exp, ast.Constant(value=None)))
            r_nodes.append(coro)
            call = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=self.sp_mod_name, ctx=ast.Load()),
                    attr='_start',
                    ctx=ast.Load()),
                args=[ast.Call(func=ast.Name(id=tmp_name, ctx=ast.Load()), args=[], keywords=[])],
                keywords=[]
            )
            r_nodes.append(ast.Expr(call))
            self._exit_context()
        return r_nodes

    def _visit_nodes(self, nodes: List[ast.AST], r_nodes: Optional[List[ast.AST]] = None) -> List[ast.AST]:
        if nodes is None:
            return None
        if r_nodes is None:
            r_nodes = []
        for node in nodes:
            r_node = self.visit(node)
            r_nodes.append(r_node)
        return r_nodes


def dump(f):
    t = ast.parse(inspect.getsource(f))
    astdump(t)
    return f


def _check_async(node):
    if isinstance(node, ast.Module):
        if len(node.body) == 1:
            d = node.body[0]
            if isinstance(d, ast.AsyncFunctionDef):
                return True
            else:
                return False
        else:
            return False
    else:
        return False


def _find_sp(decorator_list: List[ast.AST], crt_frame) -> types.ModuleType:
    assert decorator_list
    for item in decorator_list:
        if isinstance(item, ast.Call):
            item = item.func
        while isinstance(item, ast.Attribute):
            item = item.value
        if isinstance(item, ast.Name):
            try:
                name_ref = _find_global(crt_frame, item.id)
                if isinstance(name_ref.value, types.ModuleType) and name_ref.value.__name__ == 'stpr':
                    return item.id
            except KeyError:
                pass
    raise RuntimeError('Cannot find sp module from decorator.')


class _MyUnparser(ast._Unparser):
    def generic_visit(self, node):
        try:
            return super().generic_visit(node)
        except:
            print('cannot visit %s' % node)
            raise


class _LineUpdater(ast.NodeVisitor):
    def __init__(self, offset: int) -> None:
        self.offset = offset

    def visit(self, node):
        if hasattr(node, 'lineno'):
            node.lineno += self.offset
        if hasattr(node, 'end_lineno'):
            node.end_lineno += self.offset
        return super().visit(node)


def _update_linenos(node: ast.AST, offset: int) -> None:
    _LineUpdater(offset).visit(node)


def _get_code(t: Tuple[object]) -> types.CodeType:
    for c in t:
        if isinstance(c, types.CodeType):
            return c
    raise RuntimeError('Code object not found in %s' % t)


def fn(*args, crt_frame=None, stpr_module_name=None, debug=False):
    """
    Decorator for Stpr functions.

    This decorator is used to enable Stpr functionality in a function or method. Specifically:

    * The decorated function becomes an async function (i.e., it behaves as if defined with
      ``async def``.
    * The Stpr concurrency primitives become available in that function.
    * Calls to other async functions or methods from the body of the decorated function will
      automatically be ``await``-ed. This includes calls to other ``stpr.fn`` decorated functions.
    * When an async iterator is used in a ``for`` statement, the ``for`` statement is
      transformed to an ``async for``.
    * When an async context manager is used in a ``with`` statement, the ``with`` statement
      becomes an ``async with`` statement.
    * Calls to synchronous functions/methods are automatically executed in a thread pool. Stpr
      may, heuristically, choose to execute some simple functions inside the asyncio event loop
      to reduce the overhead of submitting to a thread pool. Multiple sequential synchronous
      function/method invocations are generally merged into a single task to be submitted to a
      thread pool.

    Stpr does most of its analysis statically. However, it is not always possible to fully determine
    the type of a variable or parameter statically. When Stpr encounters such a situation, it wraps
    calls into utility functions that dynamically route execution to either the async event loop or
    to a thread pool.
    """

    if crt_frame is None:
        crt_frame = inspect.currentframe()

    def inner(f):
        nonlocal debug, crt_frame, stpr_module_name
        if DEBUG:
            debug = True
        if debug:
            _print('Instrumenting %s' % f, Color.BLUE)
        if not hasattr(f, '__SP_CC'):
            lineno = f.__code__.co_firstlineno
            t = ast.parse(textwrap.dedent(inspect.getsource(f)))
            if stpr_module_name is None:
                stpr_module_name = _find_sp(t.body[0].decorator_list, crt_frame.f_back)

            if debug:
                _print('Before instrumentation', Color.BLUE)
                astdump(t)
            t2 = _transform(t, crt_frame.f_back, stpr_module_name)
            nlines = t2.body[0].end_lineno - t2.body[0].lineno
            t2.body[0].lineno = lineno
            t2.body[0].end_lineno = lineno + nlines
            ast.fix_missing_locations(t2)
            _update_linenos(t2, lineno - 1)
            if debug:
                _print('After instrumentation', Color.BLUE)
                astdump(t2)
            up = _MyUnparser()
            if debug:
                print(up.visit(t2))
            try:
                code = compile(t2, inspect.getfile(f), 'exec')
            except Exception as ex:
                print(ex)
                astdump(t2)
                raise
            return types.FunctionType(_get_code(code.co_consts), f.__globals__)
            f.__SP_CC = True
        return f

    if len(args) > 1:
        raise TypeError()
    if len(args) == 0:
        return inner
    else:
        return inner(args[0])


class seq:
    """
    Runs statements sequentially.

    Syntax:

    .. code-block:: Python

        with stpr.seq:
            stmt1
            stmt2
            ...
            stmtN

    Runs statements in sequence: ``stmt1`` will be run first, followed by ``stmt2`` and so on.

    .. uml::
        :align: center

        @startuml
        <style>
            activityDiagram {
                .blank {
                    LineColor #ffffff
                    BackgroundColor #ffffff
                }
            }
        </style>
        start
        fork
            :stmt1;
            :stmt2;
            :...; <<blank>>
            :stmtN;
        end fork
        stop
        @enduml

    When an exception is raised by the currently executing statement, the execution of the remaining
    statements is aborted and ``stpr.seq`` re-raises the exception raised by the statement.

    Running statements sequentially is the default in a Stpr function so including a ``stpr.seq``
    is redundant if all statements in a function are to be run sequentially. The typical usage
    scenario of ``stpr.seq`` is when it is nested inside :py:func:`stpr.parallel` or other similar
    concurrency primitives.
    """

    def __enter__(self):
        debug_print('%0.4f enter %s' % (_ts(), self), Color.GREEN)

    def __exit__(self, exc_type, exc_val, exc_tb):
        debug_print('%0.4f exit %s (exc: %s)' % (_ts(), self, exc_val), Color.GREEN)

    @classmethod
    def _transformer(cls) -> Callable:
        return _Transformer._seq


class parallel:
    """
    Runs statements in parallel.

    Syntax:

    .. code-block:: Python

        with stpr.parallel:
            stmt1
            stmt2
            ...
            stmtN

    Runs ``stmt1``, ``stmt2``, ..., and ``stmtN`` in parallel and waits for all of them to complete.

    .. uml::
        :align: center

        @startuml
        <style>
            activityDiagram {
                .blank {
                    LineColor #ffffff
                    BackgroundColor #ffffff
                }
            }
        </style>
        start
        fork
            :stmt1;
        fork again
            :stmt2;
        fork again
            -[#white]->
            :...; <<blank>>
            -[#white]->
        fork again
            :stmtN;
        end fork
        stop
        @enduml

    If one or more statements raise an exception, the remaining running statements are canceled and
    the first detected exception is re-raised. This is in contrast with
    :py:class:`asyncio.TaskGroup`, which raises an exception that combines all detected exceptions
    from its tasks.

    An alternative syntax is to use ``stpr.parallel`` as a function:

    .. code-block:: Python

        r1, r2, ..., rN = stpr.parallel(f1(), f2(), ..., fN())

    In this case, ``f1``, ``f2``, ..., and ``fN`` will be executed in parallel and their return
    values assigned, respectively, to ``r1``, ``r2``, ..., and ``rN``.
    """
    def __init__(self, *args):
        self._tg = asyncio.TaskGroup()

    async def __aenter__(self):
        debug_print('%0.4f enter %s' % (_ts(), self), Color.GREEN)
        await self._tg.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        debug_print('%0.4f exit %s (exc: %s)' % (_ts(), self, exc_val), Color.GREEN)
        try:
            await self._tg.__aexit__(exc_type, exc_val, exc_tb)
        except ExceptionGroup as eg:
            raise eg.exceptions[0] from None

    def _start(self, coro) -> Task:
        return self._tg.create_task(coro)

    @classmethod
    def _transformer(cls) -> Callable:
        return _Transformer._parallel

    @staticmethod
    async def _fn(*coros) -> Tuple[...]:
        tasks = []
        async with parallel() as ctx:
            for coro in coros:
                tasks.append(ctx._start(coro))
        r = tuple([task.result() for task in tasks])
        debug_print(f'r: {r}', Color.BRIGHT_RED)
        return r


class _RaceTaskGroup(asyncio.TaskGroup):
    def __init__(self):
        super().__init__()
        self._task_done = False
        self._result = None

    def _on_task_done(self, task):
        debug_print(f'task_done: {task}', Color.GREEN)
        super()._on_task_done(task)
        if self._task_done:
            return
        self._task_done = True
        self._result = task.result()
        debug_print(f'result: {self._result}', Color.GREEN)
        self._abort()


class race:
    """
    Races a set of functions.

    Syntax:

    .. code-block:: Python

        r = stpr.race(f1(), f2(), ..., fN())

    Functions ``f1``, ``f2``, ..., and ``fN`` are executed in parallel. The first function to
    complete successfully wins the race and its return value is returned by ``stpr.race``. The
    execution of the remaining functions is then canceled. If any function raises an exception
    after the first function completes, the exception is ignored. If one or more functions raise an
    exception before any function completes successfully, ``stpr.race`` re-raises that exception
    and cancels the execution of the remaining functions.
    """
    @staticmethod
    async def _fn(*coros) -> object:
        tasks = []
        tg = _RaceTaskGroup()
        async with tg:
            for coro in coros:
                tasks.append(tg.create_task(coro))
        r = tg._result
        debug_print(f'r: {r}', Color.BRIGHT_RED)
        return r


class parallelFor:
    """
    Iterates over a sequence in parallel.

    Syntax:

    .. code-block:: Python

        with stpr.parallelFor(iterable) as var:
            stmt1
            stmt2
            ...
            stmtN

    For each value produced by the ``iterable``, ``var`` is assigned that value and the statements
    ``stmt1``, ``stmt2``, ..., and ``stmtN`` are executed sequentially. The execution of each
    iteration is started as soon as the iterable produces the corresponding value. If the
    ``iterable`` produces all its values at once and there is a total of ``M`` values, the
    following diagram applies:

    .. uml::
        :align: center

        @startuml
        <style>
            activityDiagram {
                .blank {
                    LineColor #ffffff
                    BackgroundColor #ffffff
                }
            }
        </style>
        start
        fork
            :var = iterable[0];
            :stmt1;
            :stmt2;
            :...; <<blank>>
            :stmtN;
        fork again
            :var = iterable[1];
            :stmt1;
            :stmt2;
            :...; <<blank>>
            :stmtN;
        fork again
            -[#white]->
            :...; <<blank>>
            -[#white]->
        fork again
            :var = iterable[M - 1];
            :stmt1;
            :stmt2;
            :...; <<blank>>
            :stmtN;
        end fork
        stop
        @enduml

    If any of the statements raise an exception in any of the iterations, ``stpr.parallelFor``
    cancels all running iterations, stops processing values from ``iterator``, and re-raises that
    exception.
    """
    def __init__(self, it: Iterable[object], maxp: Optional[int] = None):
        """
        :param it: The iterable to iterate over.
        :param maxp: If specified, at most ``maxp`` parallel iterations will be active at any given
            time.
        """
        pass

    @classmethod
    def _transformer(cls) -> Callable:
        return _Transformer._parallelFor


class fork:
    """
    Runs a statement asynchronously.
    """
    def __init__(self):
        pass

    def __enter__(self):
        debug_print('%0.4f enter %s' % (_ts(), self), Color.BLUE)

    def __exit__(self, exc_type, exc_val, exc_tb):
        debug_print('%0.4f exit %s' % (_ts(), self))

    @classmethod
    def _transformer(cls) -> Callable:
        return _Transformer._fork


_SP_CMS = [seq, parallel, parallelFor, fork]

_SP_FNS = [parallel, race]
