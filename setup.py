from setuptools import setup, find_packages

with open('src/stpr/version.py') as f:
    exec(f.read())


if __name__ == '__main__':
    with open('requirements.txt') as f:
        install_requires = f.readlines()

    setup(
        name='stpr',
        version=VERSION,

        description='''The Stpr structured parallelism library.''',

        author='',
        author_email='hategan@mcs.anl.gov',

        url='https://github.com/hategan/stpr',

        classifiers=[
            'Programming Language :: Python :: 3',
            'License :: OSI Approved :: MIT License',
        ],


        package_dir={'': 'src'},

        package_data={
            '': ['README.md', 'LICENSE'],
            'stpr': ['py.typed']
        },

        scripts=[],

        entry_points={
        },

        install_requires=install_requires,
        python_requires='>=3.8'
    )
