from setuptools import setup, find_packages


INSTALL_REQUIREMENTS = [
    'numpy',
    'pandas',
    'pytest',
    'scikit-learn',
    'matplotlib',
    'flask',
    'flask_restful',
    'pylint',
    'keras',
    'tensorflow',
    'opencv-contrib-python',
    'absl-py',
    'gast',
    'astor',
    'pyyaml',
    'scipy',
    'h5py'
    ]
setup(name='roadseg',
      packages=find_packages(),
      version='0.0.1dev1',
      entry_points={
          'console_scripts': ['marabou-train=roadseg.scripts.train_chain:main',
                              'marabou-eval=roadseg.scripts.eval_chain:main',
                              'marabou-rest-api=roadseg.scripts.serve_rest:main']
      },
      install_requires=INSTALL_REQUIREMENTS,
      package_data={},
      include_package_data=True)