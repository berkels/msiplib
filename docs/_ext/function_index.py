import pkgutil
import inspect
import importlib

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.parsers.rst import directives

import msiplib


class FunctionIndex(Directive):

    option_spec = {
        'showmodules': directives.unchanged
    }

    def run(self):
        # Compile a list of all functions
        functions = []

        for importer, modname, ispkg in pkgutil.walk_packages(path=msiplib.__path__,
                                                              prefix=msiplib.__name__+'.',
                                                              onerror=lambda x: None):
            m = importlib.import_module(modname)

            funcs = inspect.getmembers(m)
            funcs = [f for f in funcs if inspect.getmodule(f[1]) == m and callable(f[1]) and not inspect.isclass(f[1])]

            functions.extend([(modname, f[0]) for f in funcs])

        funcs = inspect.getmembers(msiplib)
        funcs = [f for f in funcs if inspect.getmodule(f[1]) == msiplib and callable(f[1]) and not inspect.isclass(f[1])]

        functions.extend([(msiplib.__name__, f[0]) for f in funcs])

        # Sort the functions alphabetically
        functions = sorted(functions, key=lambda f: f[1])

        # Finally add the functions to a bullet_list
        l = nodes.bullet_list()
        for f in functions:
            li = nodes.list_item()
            li_contents = nodes.inline()
            li_contents += nodes.reference(refuri="../msiplib/" + f[0] + '.html#' + f[0] + '.' + f[1], text=f[1])
            if 'showmodules' in self.options:
                li_contents += nodes.inline(text=" (in " + f[0] + ")")
            li += li_contents
            l += li

        return [l]


def setup(app):
    app.add_directive("functionindex", FunctionIndex)
