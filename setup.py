from setuptools import setup, find_packages

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
	name = 'qfipy',
	version = '1.0.0',
	long_description = long_description,
	author = 'Apostolos Kouzoukos',
	author_email = 'kouzoukos97@gmail.com',
	url = 'https://github.com/kouzapo/QFiPy',
	packages = ['qfipy'],
    package_dir = {'qfipy': 'qfipy'},
	package_data = {'qfipy': ['data/symbols_files/*.dat']},
	classifiers=[
        'Programming Language :: Python',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: Unix',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        ],
    install_requires = ['numpy', 'scipy', 'pandas', 'pandas-datareader']
)
