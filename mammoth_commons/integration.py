# autoflake: skip_file
from typing import get_type_hints, Dict, List, get_origin, get_args, Union, Literal
from functools import wraps
import subprocess
import inspect
import sys
import os

_default_python = "3.12"
_default_packages = ()  # appended to ["mammoth_commons[deployment]"]


def install_package(package, record_file=".cache/installed.txt"):
    command_line = f"{sys.executable} -m pip install {package}"
    os.makedirs(os.path.dirname(record_file), exist_ok=True)
    if not os.path.exists(record_file):
        open(record_file, "w").close()
    with open(record_file, "r") as f:
        installed = {line.strip() for line in f}
    if command_line not in installed:
        try:
            print(f"Installing: {package}")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install"] + package.split(" "),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for line in result.stdout.splitlines():
                if not line.startswith("Requirement already satisfied:"):
                    print(line)
            if result.returncode != 0:
                raise subprocess.CalledProcessError(
                    result.returncode, result.args, output=result.stdout
                )
        except subprocess.CalledProcessError as e:
            raise Exception(f"Failed to install: {package}: {e.output}")
        with open(record_file, "a") as f:
            f.write(command_line + "\n")
    return True


def unpack_optionals(arg_type):
    # Check if the type is Optional (which is the same as Union[type, None])
    if get_origin(arg_type) is Union and type(None) in get_args(arg_type):
        # Return the internal type (excluding None)
        return [arg for arg in get_args(arg_type) if arg is not type(None)][0]
    return arg_type


def fixed_version(library):
    if library == "numpy":
        return "numpy>=2.0.0"
    if library == "skl2onnx":
        return "skl2onnx>=1.19.1"
    if library == "fairsearch":
        return "git+https://github.com/maniospas/fairsearch-fair-python.git"
    if library == "mmm-fair-cli":
        return "mmm-fair-cli --ignore-requires-python"
    #    return "git+https://github.com/onnx/sklearn-onnx.git@7322ef5753ec9ed5774c83b3809a75deaa0aee29"
    return library


def _path(method):
    running_path = os.path.abspath(os.getcwd()).lower()
    method_path = os.path.abspath(inspect.getfile(method)).lower()
    assert method_path.startswith(
        running_path
    ), f"Running path is not a super-path of the path of module {method.__name__}:\nRunning path : {running_path}\nModule path: {method_path}\nHOW TO FIX:-\n- If you are running tests, create a launch configuration from the top level of mammoth.\n- If you are building, change the console's folder (CD) to the top directory of mammoth."
    method_path = method_path[len(running_path) :]
    method_path = os.path.join(".", *method_path.split(os.sep)[:-1])
    return method_path


def _class_to_name(arg_type):
    return arg_type.__name__


def _class_to_outputs(arg_type):
    return list(
        set(
            [_class_to_name(arg_type)]
            + [_class_to_name(base) for base in arg_type.__bases__]
        )
    )


class Options:
    def __init__(self, *args):
        self.values = list(args)

    def __call__(self):
        # the existence of this method introduces compatibility with typehints for Python 3.10 or earlier
        pass


def metric(
    namespace, version, python=_default_python, packages=_default_packages, logo=None
):
    # if "numpy" not in packages:
    #    packages = ["numpy"] + list(
    #        packages
    #    )  # this forces the numpy installation to be fixed
    packages = [fixed_version(package) for package in packages]
    from mammoth_commons import custom_kfp
    import yaml

    def wrapper(method):
        if logo:
            method.__doc__ = logo + method.__doc__

        @wraps(method)
        def wrapper_with_installation_outiside_kfp(*args, **kwargs):
            from mammoth_commons.externals import notify_progress, notify_end

            for i, package in enumerate(packages):
                install_package(package)
                notify_progress(
                    i / len(packages),
                    "Verifying and installing dependencies: " + package,
                )
            notify_end()
            return method(*args, **kwargs)

        # prepare the kfp wrapper given decorator arguments
        name = method.__name__  # will use this as the component id
        base_image = f"python:{python}-slim-bullseye"
        target_image = f"{namespace}/{name}:{version}"
        kfp_wrapper = lambda func: custom_kfp.custom_create_component_from_func(
            func,
            true_func=method,
            base_image=base_image,
            target_image=target_image,
            packages_to_install=["mammoth_commons[deployment]"] + list(packages),
        )

        # find signature and check that we can obtain the integration type from the returned type
        signature = inspect.signature(method)
        return_type = signature.return_annotation
        if return_type is inspect.Signature.empty:
            raise Exception(f"The metric {name} must declare a return type")
        if not hasattr(return_type, "integration"):
            raise Exception(
                f"Missing static field in the return type of {name}: {return_type.__name__}.integration"
            )

        # keep type hint names, keeping default kwargs (these will be kwarg parameters)
        type_hints = get_type_hints(method)
        defaults = dict()
        input_types = list()
        options = ""
        for pname, parameter in signature.parameters.items():
            if (
                pname == "sensitive"
            ):  # do not consider the sensitive attributes for component types
                continue
            arg_type = unpack_optionals(type_hints.get(pname, parameter.annotation))
            if isinstance(arg_type, Options):
                arg_type.__name__ = pname
                options += "\n        " + pname + ": "
                options += ", ".join(arg_type.values)
                arg_type = str
            else:
                origin = get_origin(arg_type)
                args = get_args(arg_type)
                if origin is Literal:
                    literal_values = [str(v) for v in args]
                    options += f"\n        {pname}: " + ", ".join(literal_values)
                    arg_type = str
                elif origin is Union and any(a is type(None) for a in args):
                    inner = [a for a in args if a is not type(None)][0]
                    inner_origin = get_origin(inner)
                    if inner_origin is Literal:
                        literal_values = [str(v) for v in get_args(inner)]
                        options += f"\n        {pname}: " + ", ".join(literal_values)
                        arg_type = str

            if parameter.default is not inspect.Parameter.empty:  # ignore kwargs
                defaults[pname] = (
                    "None" if parameter.default is None else parameter.default
                )
                continue
            if pname not in ["dataset", "model"]:
                raise Exception(
                    f"Only `dataset`, `model`, `sensitive` and positional arguments are supported for metrics: provide a default (e.g., None) for `{pname}`"
                )
            if arg_type is inspect.Signature.empty:
                raise Exception(
                    f"Add a type annotation in method {name} for the argument `{pname}`"
                )
            input_types.append(_class_to_name(arg_type))
            # print(f"Argument: {pname}, Type: {arg_type.__name__}")
        if len(input_types) != 2:
            raise Exception(
                "Your metric should have both a `dataset` and `model` arguments"
            )

        if options:
            method.__doc__ += "\n    Options:" + options
            wrapper_with_installation_outiside_kfp.__doc__ += "\n    Options:" + options

        # create component_metadata/{name}_meta.yaml
        metadata = {
            "id": name,
            "name": " ".join(name.split("_")),
            "description": method.__doc__,
            "parameter_info": (
                "No parameters needed."
                if not defaults
                else "Some parameters are needed."
            ),
            "component_type": "METRIC",
            "input_types": input_types,
            "parameter_default": defaults,
            "output_types": [],  # no kfp output, the data are exported when running the metric
        }
        if not os.path.exists(_path(method) + "/component_metadata/"):
            os.makedirs(_path(method) + "/component_metadata/")
        with open(f"{_path(method)}/component_metadata/{name}_meta.yaml", "w") as file:
            yaml.dump(metadata, file, sort_keys=False)

        exec_context = globals().copy()
        exec_context.update(locals())
        param_name = name + "__params"
        # create the kfp method to be wrapped
        exec(
            f"""
from kfp import dsl
import pickle

def kfp_method(
    model: dsl.Input[dsl.Model],
    dataset: dsl.Input[dsl.Dataset],
    output: dsl.Output[{return_type.integration}],
    sensitive: List[str],
    {param_name}: Dict[str, any] = defaults
):
    parameters = {param_name}
    """
            + """
    with open(dataset.path, "rb") as f:
        dataset_instance = pickle.load(f)
    with open(model.path, "rb") as f:
        model_instance = pickle.load(f)
    parameters = {
        **defaults,
        **parameters,
    }  # insert missing defaults into parameters (TODO: maybe this is not needed)
    parameters = {
        k: None if isinstance(v, str) and v == "None" else v
        for k, v in parameters.items()
    }
    ret = method(dataset_instance, model_instance, sensitive, **parameters)
    assert isinstance(ret, return_type)
    ret.export(output)
        """,
            exec_context,
        )

        kfp_method = exec_context["kfp_method"]

        # rename the kfp_method so that kfp will create an appropriate name for it
        kfp_method.__name__ = name
        kfp_method.__module__ = method.__module__
        kfp_method.__mammoth_wrapped__ = wrapper_with_installation_outiside_kfp
        # return the wrapped kfp method
        return kfp_wrapper(kfp_method)

    return wrapper


def loader(
    namespace,
    version,
    ltype=None,
    python=_default_python,
    packages=_default_packages,
    logo=None,
):
    packages = [fixed_version(package) for package in packages]
    from mammoth_commons import custom_kfp
    import yaml

    def wrapper(method, ltype):
        if logo:
            method.__doc__ = logo + method.__doc__

        @wraps(method)
        def wrapper_with_installation_outiside_kfp(*args, **kwargs):
            from mammoth_commons.externals import notify_progress, notify_end

            original_doc = method.__doc__

            for i, package in enumerate(packages):
                notify_progress(
                    i / len(packages),
                    "Verifying and installing dependencies: " + package,
                )
                install_package(package)
            notify_progress(0.99, "Running module...")
            ret = method(*args, **kwargs)
            if not ret.description:
                param_description = ""
                for k, v in kwargs.items():
                    param_description += (
                        "<b>" + k.lower().replace("_", " ") + "</b>: " + str(v) + "<br>"
                    )
                ret.description = param_description + original_doc
            notify_end()
            return ret

        # Prepare the KFP wrapper given decorator arguments
        name = method.__name__  # Will use this as the component id
        if ltype is None:
            if "data" in name.lower():
                ltype = "LOADER_DATA"
                if "model" in name.lower():
                    raise Exception(
                        "You can't have both `data` and `model` as part of your loader's name when its return type is not explicitly declared."
                    )
            elif "model" in name.lower():
                ltype = "LOADER_MODEL"
            else:
                raise Exception(
                    "Either `data` or `model` should be part of your loader's name when its return type is not explicitly declared."
                )

        base_image = f"python:{python}-slim-bullseye"
        target_image = f"{namespace}/{name}:{version}"
        kfp_wrapper = lambda func: custom_kfp.custom_create_component_from_func(
            func,
            true_func=method,
            base_image=base_image,
            target_image=target_image,
            packages_to_install=["mammoth_commons[deployment]"] + list(packages),
        )

        # Find signature and check that we can obtain the integration type from the returned type
        signature = inspect.signature(method)
        return_type = signature.return_annotation
        if return_type is inspect.Signature.empty:
            raise Exception(f"The loader {name} must declare a return type")
        if not hasattr(return_type, "integration"):
            raise Exception(
                f"Missing static field in the return type of {name}: {return_type.__name__}.integration"
            )
        if return_type.integration is inspect.Signature.empty:
            raise Exception(
                f"The loader {name} must declare a return type which is type-hinted"
            )

        # Keep type hint names, keeping default kwargs (these will be kwarg parameters)
        type_hints = get_type_hints(method)
        defaults = dict()
        options = ""
        for pname, parameter in signature.parameters.items():
            arg_type = unpack_optionals(type_hints.get(pname, parameter.annotation))
            if isinstance(arg_type, Options):
                arg_type.__name__ = pname
                options += "\n        " + pname + ": "
                options += ", ".join(arg_type.values)
                arg_type = str  # Assuming options are string-based; adjust as needed
            else:
                origin = get_origin(arg_type)
                args = get_args(arg_type)
                if origin is Literal:
                    literal_values = [str(v) for v in args]
                    options += f"\n        {pname}: " + ", ".join(literal_values)
                    arg_type = str
                elif origin is Union and any(a is type(None) for a in args):
                    inner = [a for a in args if a is not type(None)][0]
                    inner_origin = get_origin(inner)
                    if inner_origin is Literal:
                        literal_values = [str(v) for v in get_args(inner)]
                        options += f"\n        {pname}: " + ", ".join(literal_values)
                        arg_type = str
            if parameter.default is not inspect.Parameter.empty:  # Ignore kwargs
                defaults[pname] = (
                    "None" if parameter.default is None else parameter.default
                )
                continue
            # Add handling for loader-specific parameters if necessary
            raise Exception(
                f"Add both a type annotation and default value in method {name} for the argument: {pname}"
            )

        original_doc = method.__doc__
        if options:
            method.__doc__ += "\n    Options:" + options
            wrapper_with_installation_outiside_kfp.__doc__ += "\n    Options:" + options

        # Create component_metadata/{name}_meta.yaml
        metadata = {
            "id": name,
            "name": " ".join(name.split("_")),
            "description": method.__doc__,
            "parameter_info": (
                "No parameters needed."
                if not defaults
                else "Some parameters are needed."
            ),
            "component_type": ltype,
            "parameter_default": defaults,
            "output_types": _class_to_outputs(return_type),
        }
        if not os.path.exists(_path(method) + "/component_metadata/"):
            os.makedirs(_path(method) + "/component_metadata/")
        with open(f"{_path(method)}/component_metadata/{name}_meta.yaml", "w") as file:
            yaml.dump(metadata, file, sort_keys=False)
        param_name = name + "__params"
        exec_context = globals().copy()
        exec_context.update(locals())

        # Create the KFP method to be wrapped
        exec(
            f"""
from kfp import dsl
import pickle

def kfp_method(
    output: dsl.Output[{return_type.integration}],
    {param_name}: Dict[str, any] = defaults,
):
    parameters = {param_name}
    original_doc = \"\"\"{original_doc}\"\"\"
    """
            + """
    parameters = {
        **defaults,
        **parameters,
    }  # Insert missing defaults into parameters (TODO: maybe this is not needed)
    parameters = {
        k: None if isinstance(v, str) and v == "None" else v
        for k, v in parameters.items()
    }
    
    ret = method(**parameters)
    assert isinstance(ret, return_type)
    if not ret.description:
        param_description = ""
        for k, v in parameters.items():
            param_description += "<b>" + k.lower().replace("_", " ") + "</b>: " + str(v) + "<br>"
        ret.description = param_description+original_doc  
    with open(output.path, "wb") as file:
        pickle.dump(ret, file)
            """,
            exec_context,
        )
        kfp_method = exec_context["kfp_method"]
        # Rename the kfp_method so that KFP will create an appropriate name for it
        kfp_method.__name__ = name
        kfp_method.__module__ = method.__module__
        kfp_method.__mammoth_wrapped__ = wrapper_with_installation_outiside_kfp

        # Return the wrapped KFP method
        return kfp_wrapper(kfp_method)

    return lambda method: wrapper(method, ltype)  # Properly pass ltype to the wrapper
