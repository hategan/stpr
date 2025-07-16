import os
from typing import List, Tuple

import subprocess
import sys

from sphinx.ext.apidoc import main


web_docs = False
if '_WEB_DOCS' in os.environ:
	web_docs = True


needs_sphinx = '1.6'

# Set Sphinx variables
master_doc = 'index'

project = u'stpr'
copyright = u''
if web_docs:
	html_theme = 'piccolo_theme'
else:
	html_theme = 'sphinx_rtd_theme'
html_favicon = 'favicon.png'
autoclass_content = 'both'
add_module_names = False
python_use_unqualified_type_names = True
autodoc_mock_imports = []
nitpick_ignore = [
    ('py:class', 'distutils.version.StrictVersion'),
    ('py:class', 'distutils.version.Version'),
    ('py:class', 'packaging.version.Version')
]

if web_docs:
    templates_path = ['../web/_templates']
    # Extra JS/CSS
    html_static_path = ['../web/_static']
    html_js_files = ['main.js']
    html_css_files = ['style.css', 'fixes.css']

    # Unfortunately sphinx-multiversion does not properly deal with
    # setting the title to the proper version. You either get some
    # default like "0.0.1" or you get whatever the current conf.py
    # sets (i.e., the latest version).
    # See, e.g., https://github.com/Holzhaus/sphinx-multiversion/issues/61
    #
    # But we already have the version selector that displays the version,
    # so we can display that where the broken version would otherwise
    # have appeared.
    html_title = 'stpr'
    # Multi-version
    smv_branch_whitelist = '^matchmeifyoucan$'
    smv_remote_whitelist = None
    smv_released_pattern = r'^\d+\.\d+\.\d+(\..*)?$'
    smv_outputdir_format = 'v/{ref.name}'
    html_context = {'doc_page': True}
else:
    html_static_path = ['./_static']
    html_css_files = ['fixes.css']


html_sidebars = {'**': ['globaltoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html']}

# Setup Sphinx extensions (and associated variables)
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.autodoc.typehints',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.graphviz',
    'sphinxcontrib.plantuml'
]

if web_docs:
    extensions.append('sphinx_multiversion')

autodoc_typehints = 'both'
autodoc_typehints_format = 'short'
autodoc_default_options = {
    'show-inheritance': True
}

cwd = os.getcwd()

plantuml = f'java -jar {cwd}/plantuml.jar'
plantuml_output_format = 'svg'


rst_prolog = """
.. |project_name| replace:: stpr
.. |pkg_name| replace:: stpr
"""

release = None
version = None
src_dir = None


def read_version(docs_dir):
    global release, version, src_dir
    src_dir = os.path.abspath(os.path.join(docs_dir, '../src'))

    sys.path.insert(0, src_dir)

    import stpr
    release = stpr.__version__
    version = release


my_dir = os.path.normpath(os.path.dirname(__file__))
read_version(my_dir)

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
}

# -- Setup for Sphinx API Docs -----------------------------------------------

# Workaround since sphinx does not automatically run apidoc before a build
# Copied from https://github.com/readthedocs/readthedocs.org/issues/1139

# run api doc
def run_apidoc(sphinx):
    return
    read_version(sphinx.srcdir)  # this sets src_dir based on the version being compiled
    output_path = os.path.join(sphinx.srcdir, '.generated')
    os.makedirs(output_path, exist_ok=True)
    generate_path = os.path.join(sphinx.srcdir, 'generate.py')
    if os.path.exists(generate_path):
        # use the generate script if it exists
        subprocess.run([sys.executable, generate_path], cwd=sphinx.srcdir, check=True, 
                       env={'PYTHONPATH': src_dir})
    else:
        main(['-f', '-t', os.path.join(my_dir, '_sphinx'), '-o', output_path, src_dir])


# The following is a hack to allow returns in numpy style doctstrings to
# not duplicate the return type specified by the normal type hints.
# This was taken from https://github.com/svenevs/elba
# It was submitted as a patch to the napoleon extension. Unfortunately,
# it was rejected by the maintainers of napoleon
# (see https://github.com/sphinx-doc/sphinx/issues/7077).
def _consume_returns_section(self) -> List[Tuple[str, str, List[str]]]:
    self._consume_empty()
    desc_lines = []
    while not self._is_section_break():
        desc_lines.append(next(self._line_iter))

    return [("", "", desc_lines)]

from sphinx.ext.napoleon.docstring import NumpyDocstring
NumpyDocstring._consume_returns_section = _consume_returns_section


# And this is for "More than one target found for cross-reference"
# See https://github.com/sphinx-doc/sphinx/issues/3866
from sphinx.domains.python import PythonDomain

class MyPythonDomain(PythonDomain):
    def find_obj(self, env, modname, classname, name, type, searchmode=0):
        """Ensures an object always resolves to the desired module if defined there."""
        orig_matches = PythonDomain.find_obj(self, env, modname, classname, name, type, searchmode)
        matches = []
        for match in orig_matches:
            match_name = match[0]
            desired_name = 'stpr.' + name.strip('.')
            if match_name == desired_name:
                matches.append(match)
                break
        if matches:
            return matches
        else:
            return orig_matches

# launch setup
def setup(app):
    app.add_domain(MyPythonDomain, override=True)
    app.connect('builder-inited', run_apidoc)
