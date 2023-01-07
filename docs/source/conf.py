# -*- coding: utf-8 -*-
import datetime

now = datetime.datetime.now()

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'AnalogVNN'
copyright = str(now.year) + ', Vivswan Shah (vivswanshah@pitt.edu)'
author = 'Vivswan Shah'
release = '0.0.1'

# -- General configuration
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    'sphinx.ext.duration',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'autoapi.extension',
]

autoapi_dirs = ['../../analogvnn']
templates_path = ['_templates']
exclude_patterns = []

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

# -- Options for HTML output
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
# html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Options for EPUB output
epub_show_urls = 'footnote'
