from setuptools import setup, Extension

# Cファイルを指定してビルドする設定
module = Extension('python', sources=['python.c'])

setup(
    name='python',
    version='1.0',
    description='Python -> C -> Assembly Bridge',
    ext_modules=[module]
)