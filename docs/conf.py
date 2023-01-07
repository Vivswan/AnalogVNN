# -*- coding: utf-8 -*-
from datetime import date

try:
    from analogvnn import __version__

    print('Version from module: {}'.format(__version__))
except Exception:
    with open('../pyproject.toml', 'r') as f:
        for i in f.readlines():
            if 'version' in i:
                __version__ = i.split('=')[1].strip().strip('"')
                break
    print('Version from toml: {}'.format(__version__))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Project information
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
project = 'AnalogVNN'
copyright = f'{date.today().year}, Vivswan Shah (vivswanshah@pitt.edu)'
author = 'Vivswan Shah'
release = __version__

# General configuration
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    'autoapi.extension',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.duration',
    'sphinx.ext.mathjax',
    'sphinx_copybutton',
    'notfound.extension',
    'sphinx_inline_tabs',
    'sphinxext.opengraph',
    'myst_parser',
]

autosummary_generate = True
# napoleon_google_docstring = True
# napoleon_include_init_with_doc = True
# napoleon_include_private_with_doc = True
# napoleon_include_special_with_doc = True
autoapi_dirs = ['../analogvnn']
autoapi_type = 'python'
autoapi_add_toctree_entry = True
autoapi_member_order = 'groupwise'

suppress_warnings = ['myst_parser''autoapi']
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    'sphinx': ('https://www.sphinx-doc.org/en/master', None),
    'markdown_it': ('https://markdown-it-py.readthedocs.io/en/latest', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'networkx': ('https://networkx.org/documentation/stable/', None),
    'tensorflow': (
        'https://www.tensorflow.org/api_docs/python',
        'https://github.com/GPflow/tensorflow-intersphinx/raw/master/tf2_py_objects.inv'
    ),
    'tensorflow_probability': (
        'https://www.tensorflow.org/probability/api_docs/python',
        'https://github.com/GPflow/tensorflow-intersphinx/raw/master/tfp_py_objects.inv'
    ),
    "torch": ("https://pytorch.org/docs/stable/", None),

}
intersphinx_disabled_domains = ['std']

# Options for HTML output
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = 'furo'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_theme_options = {
    'light_logo': 'analogvnn-logo-wide-white.svg',
    'dark_logo': 'analogvnn-logo-wide-black.svg',
    'source_repository': 'https://github.com/Vivswan/AnalogVNN',
    'source_branch': 'main',
    'source_directory': 'docs/',
}
# html_logo = '_static/analogvnn-logo-wide-black.svg'
html_favicon = '_static/analogvnn-logo-square-black.svg'
html_title = f'AnalogVNN {release}'
language = 'en'

# Options for EPUB output
epub_show_urls = 'footnote'
