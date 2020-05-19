from hyperopt.pyll.base import Apply as hp_apply

__all__ = [
    "unflattened_params",
    "updated_params",
]

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

def updated_params(base, additions):
    params = dict(base)
    for key, value in additions.items():
        if isinstance(value, dict):
            value = updated_params(params.get(key, {}), value)
        params[key] = value

    return params

