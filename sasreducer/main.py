#!/usr/bin/python3

from sasreducer.worker import Worker
import sys

if __name__ == '__main__':

  c = Worker(sys.argv[1:])
  c.work()
