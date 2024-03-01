# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from griffe import Attribute
from griffe import Class
from griffe import Docstring
from griffe import Extension
from griffe import Object
from griffe import ObjectNode
from griffe import dynamic_import
from griffe import get_logger

if TYPE_CHECKING:
    import ast

    from griffe.enumerations import Parser

logger = get_logger(__name__)


class DocstringInheritance(Extension):
    """Inherit docstrings when the package docstring-inheritance is used."""

    __parser: Parser | None = None
    """The docstring parser."""

    __parser_options: ClassVar[dict[str, Any]] = {}
    """The docstring parser options."""

    def on_class_members(self, node: ast.AST | ObjectNode, cls: Class) -> None:
        if isinstance(node, ObjectNode):
            # Skip runtime objects, their docstrings are already OK.
            return

        runtime_cls = self.__import_dynamically(cls)

        if not self.__has_docstring_inheritance(runtime_cls):
            return

        # Inherit the class docstring.
        self.__set_docstring(cls, runtime_cls)

        # Inherit the methods docstrings.
        for member in cls.members.values():
            if not isinstance(member, Attribute):
                runtime_obj = self.__import_dynamically(member)
                self.__set_docstring(member, runtime_obj)

    @staticmethod
    def __import_dynamically(obj: Object) -> Any:
        """Import dynamically and return an object."""
        try:
            return dynamic_import(obj.path)
        except ImportError:
            logger.debug("Could not get dynamic docstring for %s", obj.path)

    @classmethod
    def __set_docstring(cls, obj: Object, runtime_obj: Any) -> None:
        """Set the docstring from a runtime object.

        Args:
            obj: The griffe object.
            runtime_obj: The runtime object.
        """
        if runtime_obj is None:
            return

        try:
            docstring = runtime_obj.__doc__
        except AttributeError:
            logger.debug("Object %s does not have a __doc__ attribute", obj.path)
            return

        if docstring is None:
            return

        # Update the object instance with the evaluated docstring.
        if obj.docstring:
            obj.docstring.value = inspect.cleandoc(docstring)
        else:
            cls.__find_parser(obj)
            obj.docstring = Docstring(
                docstring,
                parent=obj,
                parser=cls.__parser,
                parser_options=cls.__parser_options,
            )

    @staticmethod
    def __has_docstring_inheritance(cls: type[Any]) -> bool:
        for base in cls.__class__.__mro__:
            if base.__name__.endswith("DocstringInheritanceMeta"):
                return True
        return False

    @classmethod
    def __find_parser(cls, obj: Object) -> None:
        """Search a docstring parser recursively from the object parents."""
        if cls.__parser is not None:
            return None
        parent = obj.parent
        parser = parent.docstring.parser
        if parser:
            cls.__parser = parser
            cls.__parser_options = parent.docstring.parser_options
        return cls.__find_parser(parent)
