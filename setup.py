# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['imlmlib']

package_data = \
{'': ['*']}

install_requires = \
['matplotlib>=3.7.1,<4.0.0',
 'numpy>=1.24.2,<2.0.0',
 'scipy>=1.10.1,<2.0.0',
 'tqdm>=4.65.0,<5.0.0']

setup_kwargs = {
    'name': 'imlmlib',
    'version': '0.0.2.dev0',
    'description': 'Utilities for parametric modeling of interaction modalities that leverage memory',
    'long_description': None,
    'author': 'jgori',
    'author_email': 'juliengori@gmail.com',
    'maintainer': None,
    'maintainer_email': None,
    'url': None,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.8,<3.12',
}


setup(**setup_kwargs)
