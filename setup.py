from setuptools import setup

setup(name='wrandai', 
      description='This is a repository for calculating the W Randomness Coefficient \
                    of a set of algorithms on a suite of Machine Learning benchmarks.',
      url='https://github.com/willleeney/wrandai',
      author='William Leeney',
      author_email='will.leeney@outlook.com',
      license='MIT',
      version='0.3.0', 
      packages=['wrandai'],
      install_requires=['scipy', 'numpy'],
      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
       ' Intended Audience :: Information Technology',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.9',
    ],

)
