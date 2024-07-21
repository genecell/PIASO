from setuptools import setup, find_packages

setup(
    name='PIASO',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # List your dependencies here
    ],
    author='Min Dai',
    author_email='dai@broadinstitute.org',
    description='Precise Integrative Analysis of Single-cell Omics',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='http://github.com/genecell/PIASO',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)

