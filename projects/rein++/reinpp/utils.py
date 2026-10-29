_done = set()

def do_once(func, workname=None):
    key = workname if workname is not None else func
    if key not in _done:
        func()
        _done.add(key)
