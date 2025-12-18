{{ objname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:

   {% block methods %}
   .. automethod:: __init__

   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
      :nosignatures:
   {% for item in methods %}
      {%- if item != "__init__" %}
      ~{{ name }}.{{ item }}
      {%- endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
