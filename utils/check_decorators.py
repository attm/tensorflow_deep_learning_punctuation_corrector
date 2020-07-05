import functools


def type_check(func):
    """
    Decorator for checking function args types. Will raise TypeError if wrong type.
    """
    @functools.wraps(func)
    def check(*args, **kwargs):
        for i in range(len(args)):
            v = args[i]
            v_name = list(func.__annotations__.keys())[i]
            v_type = list(func.__annotations__.values())[i]
            error_msg = "TypeError in {0}, expected variable {1} of type {2}, but got type {3} instead.".format(func.__name__,
                                                                                                                str(v_name),
                                                                                                                str(v_type),
                                                                                                                str(type(v)) )
            if not isinstance(v, v_type):
                raise TypeError(error_msg)
        return func(*args, **kwargs)
    return check