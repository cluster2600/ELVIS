'''
Copyright (c) 2021 Apple Inc. All rights reserved.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import code
import sys


def main(_):
  code.interact()
  return 0


if __name__ == '__main__':
  sys.exit(main(sys.argv))
