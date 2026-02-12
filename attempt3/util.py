def flatten_once(lst):
    # this is interpreted as first for is outer, second for is inner
    return [
        item
        for element in lst
        for item in (element if isinstance(element, list) else [element])
    ]