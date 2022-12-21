import pkgutil
import inspect
import importlib

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.parsers.rst import directives

import msiplib


class ClassIndex(Directive):

    option_spec = {
        'showmodules': directives.unchanged
    }

    def run(self):
        # Compile a list of all classes
        classes = []

        for importer, modname, ispkg in pkgutil.walk_packages(path=msiplib.__path__,
                                                              prefix=msiplib.__name__+'.',
                                                              onerror=lambda x: None):
            m = importlib.import_module(modname)

            mclasses = inspect.getmembers(m, inspect.isclass)
            mclasses = [c for c in mclasses if inspect.getmodule(c[1]) == m]

            classes.extend([(modname, c[0]) for c in mclasses])

        mclasses = inspect.getmembers(msiplib, inspect.isclass)
        mclasses = [c for c in mclasses if inspect.getmodule(c[1]) == msiplib]

        classes.extend([(msiplib.__name__, c[0]) for c in mclasses])

        # Sort the classes alphabetically
        classes = sorted(classes, key=lambda c: c[1])

        # Finally add the classes to a bullet_list
        l = nodes.bullet_list()
        for c in classes:
            li = nodes.list_item()
            li_contents = nodes.inline()
            li_contents += nodes.reference(refuri="../msiplib/" + c[0] + '.html#' + c[0] + '.' + c[1], text=c[1])
            if 'showmodules' in self.options:
                li_contents += nodes.inline(text=" (in " + c[0] + ")")
            li += li_contents
            l += li

        return [l]


def setup(app):
    app.add_directive("classindex", ClassIndex)
