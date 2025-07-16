import os
from typing import List, Tuple

import subprocess
import sys

from sphinx.ext.apidoc import main


needs_sphinx = '1.6'

# Set Sphinx variables
master_doc = 'index'

project = u'stpr'
copyright = u''
html_theme = 'piccolo_theme'
html_favicon = 'favicon.png'

templates_path = ['_templates']

#html_title = "stpr"
html_logo = './_static/logo-large.png'

# Extra JS/CSS
html_static_path = ['_static']
html_js_files = ['main.js']
html_css_files = ['style.css', 'fixes.css']

html_context = {'doc_page': False}
