from setuptools import find_packages, setup


with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='contrastors',
    version="0.0.1",
    description='Contrastors',
    url='https://github.com/nomic-ai/contrastors',
    author='Nomic AI',
    author_email='zach@nomic.ai',
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    package_dir={'': 'src'},
    packages=find_packages(where='contrastors'),
    install_requires=requirements,
    extras_require={
        "eval": ["openai", "tiktoken", "mteb[beir]", "multiprocess==0.70.15"],
        "dev": ["pytest", "black", "isort"],
    },
    include_package_data=True,
    python_requires='>=3.7',
)
