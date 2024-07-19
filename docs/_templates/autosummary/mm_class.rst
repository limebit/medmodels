{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:

   {% block attributes_summary %}
   {% if attributes %}
   .. rubric:: Attributes Summary

   .. autosummary::
   {% for item in attributes %}
      {{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block methods_summary %}
   {% if methods %}
   .. rubric:: Methods Summary

   .. autosummary::
   {% for item in methods %}
      {{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes_documentation %}
   {% if attributes %}
   .. rubric:: Attributes Documentation

   {% for item in attributes %}
   .. autoattribute:: {{ name }}.{{ item }}
      :noindex:
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block methods_documentation %}
   {% if methods %}
   .. rubric:: Methods Documentation

   {% for item in methods %}
   .. automethod:: {{ name }}.{{ item }}
      :noindex:
   {%- endfor %}
   {% endif %}
   {% endblock %}
