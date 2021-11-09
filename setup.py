from setuptools import setup

modules = ['callbacks','netbase']
setup(name='skorch_extra',
      version='0.4',
      description='Extra tools when using skorch',
      url='https://github.com/Lucashsmello/skorch_extra',
      author='Lucas Mello',
      author_email='lucashsmello@gmail.com',
      license='new BSD 3-Clause',
      py_modules=["skorch_extra.%s" % m for m in modules],
      install_requires=[
          "skorch",
          "tensorboard",
      ],
      zip_safe=False)
