# -*- coding: utf-8 -*-
from datetime import date

from analogvnn import __version__

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
    'myst_parser',
    'sphinx.ext.duration',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx_copybutton',
    'notfound.extension',
    'autoapi.extension',
]

autoapi_dirs = ['../../analogvnn']
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3.7', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master', None),
    'markdown_it': ('https://markdown-it-py.readthedocs.io/en/latest', None),
}
intersphinx_disabled_domains = ['std']

# Options for HTML output
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = 'furo'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_logo = '_static/analogvnn-logo-wide-black.svg'
html_favicon = '_static/analogvnn-logo-square-black.svg'
html_title = f'AnalogVNN {release}'
language = 'en'

# Options for EPUB output
epub_show_urls = 'footnote'
