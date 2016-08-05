from functools import update_wrapper
from matrix import matrix
def decorator(d):
    """Make function d a decorator: d wraps function fn."""
    def _d(fn):
        return update_wrapper(d(fn), fn)
    update_wrapper(_d, d)
    return _d

decorator = decorator(decorator)

@decorator
def memo(f):
    """Decorator that caches the return value for each call to f(args).
    Then when called again with same args, we can just look it up."""
    cache = {}
    def _f(*args):
        try:
            return cache[args]
        except KeyError:
            cache[args] = result = f(*args)
            return result
        except TypeError:
            # some element of args can't be a dict key
            return f(args)
    return _f

@memo
def identity_matrix(n):
    res = matrix([[]])
    res.identity(n)
    return res
