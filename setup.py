from distutils.core import setup

setup(name='rad_motor',
      version='1.0.0',
      packages=[
          'rad_motor',
          'rad_motor/electromagnetics',
          'rad_motor/materials',
          'rad_motor/sizing', 
          'rad_motor/thermal'
      ],

      install_requires=[
        'openmdao>=2.0.0',
      ]
)
