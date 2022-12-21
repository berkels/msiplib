=={{ '=' * node.name|length }}==
``{{ node.name }}``
=={{ '=' * node.name|length }}==
.. highlight:: python
   :linenothreshold: 5

.. automodule:: {{ node.name }}
   :members:
   :undoc-members:
   :show-inheritance:

   ==============
   Module summary
   ==============
   .. ifconfig:: has_classes_{{ node.name|replace(".", "_") }}

      .. rubric:: Classes

      .. extautosummary:: {{ node.name }}
         :classes:

   .. ifconfig:: has_functions_{{ node.name|replace(".", "_") }}

      .. rubric:: Functions

      .. extautosummary:: {{ node.name }}
         :functions:

   .. ifconfig:: not(has_classes_{{ node.name|replace(".", "_") }} or has_functions_{{ node.name|replace(".", "_") }})

      This module does not contain any classes or functions.

   ====================
   Detailed description
   ====================

   .. ifconfig:: not(has_classes_{{ node.name|replace(".", "_") }} or has_functions_{{ node.name|replace(".", "_") }})

      This module does not contain any classes or functions.

   .. contents::
      :local:
{##}
{%- block modules -%}
{%- if subnodes %}

Submodules
==========

.. toctree::
{% for item in subnodes %}
   {{ item.name }}
{%- endfor %}
{##}
{%- endif -%}
{%- endblock -%}
{##}
.. currentmodule:: {{ node.name }}
{##}
{%- block functions -%}
{%- if node.functions %}

Functions
=========

{% for item, obj in node.functions.items() -%}
- :py:func:`{{ item }}`:
  {{ obj|summary }}

{% endfor -%}

{% for item in node.functions %}
.. autofunction:: {{ item }}
{##}
{%- endfor -%}
{%- endif -%}
{%- endblock -%}

{%- block classes -%}
{%- if node.classes %}

Classes
=======

{% for item, obj in node.classes.items() -%}
- :py:class:`{{ item }}`:
  {{ obj|summary }}

{% endfor -%}

{% for item in node.classes %}
.. autoclass:: {{ item }}
   :members:

   .. rubric:: Inheritance
   .. inheritance-diagram:: {{ item }}
      :parts: 1
{##}
{%- endfor -%}
{%- endif -%}
{%- endblock -%}

{%- block exceptions -%}
{%- if node.exceptions %}

Exceptions
==========

{% for item, obj in node.exceptions.items() -%}
- :py:exc:`{{ item }}`:
  {{ obj|summary }}

{% endfor -%}

{% for item in node.exceptions %}
.. autoexception:: {{ item }}

   .. rubric:: Inheritance
   .. inheritance-diagram:: {{ item }}
      :parts: 1
{##}
{%- endfor -%}
{%- endif -%}
{%- endblock -%}

{%- block variables -%}
{%- if node.variables %}

Variables
=========

{% for item, obj in node.variables.items() -%}
- :py:data:`{{ item }}`
{% endfor -%}

{% for item, obj in node.variables.items() %}
.. autodata:: {{ item }}
   :annotation:

   .. code-block:: text

      {{ obj|pprint|indent(6) }}
{##}
{%- endfor -%}
{%- endif -%}
{%- endblock -%}

