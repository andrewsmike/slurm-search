from pprint import pformat

from hyperopt.pyll.base import Apply as hp_apply

__all__ = [
    "mapped_params",
    "flattened_params",
    "params_from_args",
    "unflattened_params",
    "unwrapped_settings",
    "updated_params",
    "params_str",
    "params_equal",
]

def unwrapped_settings(settings):
    return {
        key: (value
              if not isinstance(value, list)
                  and len(value) == 0
              else value[0])
        for key, value in settings.items()
    }

def mapped_params(mapper_func, data, prefix=None):
    prefix = prefix or []
    if isinstance(data, dict):
        return {
            key: mapped_params(mapper_func, value, prefix + [key])
            for key, value in sorted(data.items())
        }
    elif isinstance(data, list):
        return [
            mapped_params(mapper_func, value, prefix + [index])
            for index, value in enumerate(data)
        ]
    elif isinstance(data, tuple):
        return tuple(
            mapped_params(mapper_func, value, prefix + [index])
            for index, value in enumerate(data)
        )
    else:
        return mapper_func(data, prefix)


def flattened_params(unflattened_params, writeback_dict=None, prefix=None, delim="/"):
    if not isinstance(unflattened_params, dict):
        return unflattened_params

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
    if "," in value:
        return [
            parsed_value(subval)
            for subval in value.split(",")
        ]
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

def params_str(params):
    return pformat(flattened_params(params))

def params_equal(params, other_params):
    a = flattened_params(params)
    b = flattened_params(other_params)
    return a.keys() == b.keys() and (
        all(
            isinstance(a[key], hp_apply) or (a[key] == b[key])
            for key in a.keys()
        )
    )
        
