import sys
import timeit

def alert_print(text, fx=True):
    print(f'\n\033[{"5;" if fx else ""}31m{text}\033[0m\n')

def calc_time(fun):
    def wrapper(*args, **kwargs):
        start = timeit.default_timer()
        ans = fun(*args, **kwargs)
        stop = timeit.default_timer()
        print('Time: ', stop - start)
        return ans
    return wrapper


def get_attr(obj):
    return [i for i in obj.__dict__.keys() if i[:1] != '_']


def init_str(cls_str, module_name=__name__):
    return getattr(sys.modules[module_name], cls_str)
