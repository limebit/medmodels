{{ fullname | escape | underline}}

.. toctree::
   :maxdepth: 1

   {% for item in all_attributes %}
   {{ fullname }}.{{ item }}
   {%- endfor %}
   {% for item in all_methods %}
   {{ fullname }}.{{ item }}
   {%- endfor %}
