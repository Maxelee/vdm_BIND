from setuptools import setup, find_packages

setup(
    name='vdm_bind',
    version='0.1.0',    
    description='Variational Diffusion Models for Baryonic Inference from N-body Data',
    url='https://github.com/mlee1/vdm_BIND',
    author='Max E. Lee',
    author_email='max.e.lee@columbia.edu',
    license='BSD 2-clause',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0',
        'lightning>=2.0',
        'numpy',
        'h5py',
        'pandas',
        'scipy',
        'matplotlib',
        'tqdm',
        'scikit-image',
        'joblib',
    ],
    extras_require={
        'dev': [
            'pytest',
            'jupyter',
            'tensorboard',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
