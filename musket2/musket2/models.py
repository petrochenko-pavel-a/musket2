from musket2 import binding_platform

module = type("Module", (binding_platform.ExtensionBase,), {})


