from setuptools import setup

setup(
    name='BTCBOT',
    version='0.1',
    license='GPL3',
    author='Aly Mahfouz',
    author_email='alykdev@gmail.com',
    description='BTCBOT Tool',
    install_requires=[
 
        #libs 

        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'statsmodels',
        'datetime',
        'pydata_google_auth',
        'keras',
        'sklearn',
        'tensorflow',
        'binance',
        'python-binance',
        'pystan==2.19.1.1'
        'fbprophet',

    ]
)