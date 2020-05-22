from hyperopt.pyll.base import Apply as hp_apply

__all__ = [
    "flattened_params",
    "params_from_args",
    "unflattened_params",
    "updated_params",
]


def flattened_params(unflattened_params, writeback_dict=None, prefix=None, delim="/"):
    prefix = prefix or []
    writeback_dict = writeback_dict if writeback_dict is not None else {}
    for key, value in unflattened_params.items():
        path = prefix + [str(key)]
        if isinstance(value, dict):
            flattened_params(value, writeback_dict=writeback_dict, prefix=path, delim=delim)
        else:
            writeback_dict[delim.join(path)] = value

    return writeback_dict

def unflattened_params(flattened_params, delim=":"):
    result = {}
    for path, value in flattened_params.items():
        path_parts = path.split(delim)
        key, rest = path_parts[0], path_parts[1:]
        result.setdefault(key, {})[delim.join(rest)] = value

    return {
        key: (
            unflattened_params(value)
            if "" not in value else
            value[""]
        )
        for key, value in result.items()
    }


def parsed_value(value):
    try:
        return int(value)
    except:
        pass

    try:
        return float(value)
    except:
        pass

    return value

def params_from_args(args):
    return {
        key.lstrip("-"): parsed_value(value)
        for arg in args
        for key, value in (arg.strip().split("="), )
    }

def updated_params(base, additions):
    params = dict(base)
    for key, value in additions.items():
        if isinstance(value, dict):
            value = updated_params(params.get(key, {}), value)
        params[key] = value

    return params

