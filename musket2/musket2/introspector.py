import inspect

def parameters(sig):
    if hasattr(sig, "original"):
        sig = getattr(sig, "original")
    cs = inspect.signature(sig)

    pars = []
    for v in cs.parameters:
        parameter = cs.parameters[v]
        p = {}
        if v == "self":
            continue
        if v == "args":
            continue
        if v == "kwargs":
            continue
        if parameter.kind == inspect._ParameterKind.POSITIONAL_OR_KEYWORD:
            p["kind"] = "any"
        if parameter.kind == inspect._ParameterKind.KEYWORD_ONLY:
            p["kind"] = "keyword"
        if parameter.kind == inspect._ParameterKind.POSITIONAL_ONLY:
            p["kind"] = "positional"
        if parameter.annotation != inspect._empty:
            p["type"] = parameter.annotation.__name__
        p["name"] = v
        if parameter.default != inspect._empty:
            p["defaultValue"] = str(parameter.default)
        pars.append(p)
    return pars


def extra_params(parameters):
    def inner(func):
        func.extra_params = parameters
        return func

    return inner  # this is the fun_obj mentioned in the above content


def get_meta(m):
    rs = {}
    if hasattr(m, "original"):
        m = getattr(m, "original")
    rs["doc"] = inspect.getdoc(m)
    if hasattr(m, "__name__"):
        rs["name"] = getattr(m, "__name__")

    if inspect.isclass(m):
        if hasattr(m, "__init__"):
            constructor = getattr(m, "__init__")
            rs["parameters"] = parameters(constructor)

    else:
        rs["parameters"] = parameters(m)
    try:
        rs["sourcefile"] = inspect.getsourcefile(m)
        rs["source"] = inspect.getsource(m)
        if hasattr(m, "extra_params"):
            prms = getattr(m, "extra_params")
            if "parameters" in rs:
                rs["parameters"] = rs["parameters"] + prms
            else:
                rs["parameters"] = prms
    except:
        pass
    return rs


blackList = {'get', 'deserialize', 'deserialize_keras_object', 'serialize', 'serialize_keras_object', 'Layer',
             'Callback', 'Optimizer'}


def instrospect(module, clazz):
    lst = dir(module)
    result = []
    for item in lst:
        if item[0] == '_':
            continue
        if item in blackList:
            continue
        attribute = getattr(module, item)

        if inspect.isclass(attribute):
            try:
                if isinstance(clazz, type) and issubclass(attribute, clazz):
                    result.append(get_meta(attribute, clazz.__name__))
            except:
                print("Can not inspect", attribute)

        if inspect.isfunction(attribute) and isinstance(clazz, str):
            result.append(get_meta(attribute, clazz))
    return result



