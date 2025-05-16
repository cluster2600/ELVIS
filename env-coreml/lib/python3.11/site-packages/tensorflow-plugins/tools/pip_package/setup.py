#
# Copyright (c) 2021 Apple Inc. All rights reserved.
#
'''TensorFlow acceleration for Mac GPUs.

Accelerate training of machine learning models with TensorFlow right on your Mac.
Install the base tensorflow and the tensorflow-metal PluggableDevice to accelerate training with Metal on Mac GPUs.

You can learn more about TensorFlow PluggableDevices [here](https://github.com/tensorflow/tensorflow/releases/tag/v2.5.0).


|tensorflow  |tensorflow-metal|MacOs| features | 
|---------|-----|-----|-----|
|v2.5 |v0.1.2|12.0+|Pluggable device        |
|v2.6 |v0.2.0|12.0+|Variable seq. length RNN|
|v2.7 |v0.3.0|12.0+|Custom op support       |
|v2.8 |v0.4.0|12.0+|RNN perf. improvements  |
|v2.9 |v0.5.0|12.1+|Distributed training    |
|v2.10|v0.6.0|12.1+||
|v2.11|v0.7.0|12.1+||
|v2.12|v0.8.0|12.1+||
|v2.13|v1.0.0|12.1+|FP16 and BF16 support|
|v2.14-v2.18|v1.1.0|12.1+||
|v2.18|v1.1.0|12.1+|Fixes compatibility with 2.18+ TF versions|

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import fnmatch
import os
import re
import sys

from setuptools import Command
from setuptools import find_packages
from setuptools import setup
from setuptools.command.install import install as InstallCommandBase
from setuptools.dist import Distribution

DOCLINES = __doc__.split('\n')

_VERSION = '1.2.0'

REQUIRED_PACKAGES = [
    # Enable this after updating the pypi repo with tensorflow-macos
    # 'tensorflow-macos >= 2.7.0',
    'wheel ~= 0.35',
    'six >= 1.15.0',
]

project_name = 'tensorflow-metal'

# pylint: disable=line-too-long
CONSOLE_SCRIPTS = [
]

TEST_PACKAGES = [
    'scipy >= 0.15.1',
]


class BinaryDistribution(Distribution):

  def has_ext_modules(self):
    return True


class InstallCommand(InstallCommandBase):
  """Override the dir where the headers go."""

  def finalize_options(self):
    ret = InstallCommandBase.finalize_options(self)
    self.install_headers = os.path.join(self.install_purelib,
                                        'tensorflow-plugins', 'include')
    return ret


class InstallHeaders(Command):
  """Override how headers are copied.

  The install_headers that comes with setuptools copies all files to
  the same directory. But we need the files to be in a specific directory
  hierarchy for -I <include_dir> to work correctly.
  """
  description = 'install C/C++ header files'

  user_options = [('install-dir=', 'd',
                   'directory to install header files to'),
                  ('force', 'f',
                   'force installation (overwrite existing files)'),
                 ]

  boolean_options = ['force']

  def initialize_options(self):
    self.install_dir = None
    self.force = 0
    self.outfiles = []

  def finalize_options(self):
    self.set_undefined_options('install',
                               ('install_headers', 'install_dir'),
                               ('force', 'force'))

  def mkdir_and_copy_file(self, header):
    install_dir = os.path.join(self.install_dir, os.path.dirname(header))
    install_dir = re.sub('/google/protobuf_archive/src', '', install_dir)

    external_header_locations = [
        'tensorflow-plugins/include/external/eigen_archive/',
        'tensorflow-plugins/include/external/com_google_absl/',
        'tensorflow-plugins/include/external/com_google_protobuf',
    ]
    for location in external_header_locations:
      if location in install_dir:
        extra_dir = install_dir.replace(location, '')
        if not os.path.exists(extra_dir):
          self.mkpath(extra_dir)
        self.copy_file(header, extra_dir)

    if not os.path.exists(install_dir):
      self.mkpath(install_dir)
    return self.copy_file(header, install_dir)

  def run(self):
    hdrs = self.distribution.headers
    if not hdrs:
      return

    self.mkpath(self.install_dir)
    for header in hdrs:
      (out, _) = self.mkdir_and_copy_file(header)
      self.outfiles.append(out)

  def get_inputs(self):
    return self.distribution.headers or []

  def get_outputs(self):
    return self.outfiles


def find_files(pattern, root):
  """Return all the files matching pattern below root dir."""
  for dirpath, _, files in os.walk(root):
    for filename in fnmatch.filter(files, pattern):
      yield os.path.join(dirpath, filename)


so_lib_paths = [
    i for i in os.listdir('.')
    if os.path.isdir(i) and fnmatch.fnmatch(i, '_solib_*')
]

print(os.listdir('.'))
matches = []
for path in so_lib_paths:
  matches.extend(
      ['../' + x for x in find_files('*', path) if '.py' not in x]
  )

EXTENSION_NAME = 'libmetal_plugin.so'

headers = (
    list(find_files('*.h', 'tensorflow-plugins/c_api/c')) +
    list(find_files('*.h', 'tensorflow-plugins/c_api/src')))

setup(
    name=project_name,
    version=_VERSION.replace('-', ''),
    description=DOCLINES[0],
    long_description='\n'.join(DOCLINES[2:]),
    long_description_content_type='text/markdown',
    url='https://developer.apple.com/metal/tensorflow-plugin/',
    download_url='',
    author='',
    author_email='',
    # Contained modules and scripts.
    packages= ['tensorflow-plugins'],
    entry_points={
        'console_scripts': CONSOLE_SCRIPTS,
    },
    headers=headers,
    install_requires=REQUIRED_PACKAGES,
    tests_require=REQUIRED_PACKAGES + TEST_PACKAGES,
    # Add in any packaged data.
    include_package_data=True,
    package_data={
        '': [
            EXTENSION_NAME, '*.so', '*.h', '*.py', '*.hpp'
        ] + matches,
    },
    zip_safe=False,
    distclass=BinaryDistribution,
    cmdclass={
        'install_headers': InstallHeaders,
        'install': InstallCommand,
    },
    # PyPI package information.
    classifiers=[
    ],
    license='MIT License. Copyright © 2020-2021 Apple Inc. All rights reserved.',
    keywords='tensorflow metal plugin',
)
