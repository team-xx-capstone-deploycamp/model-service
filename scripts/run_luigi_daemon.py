#!/usr/bin/env python3
import luigi
import os
from luigi.daemon import Daemon

if __name__ == '__main__':
    # Set Luigi config path
    os.environ['LUIGI_CONFIG_PATH'] = 'config/luigi.cfg'

    # Start Luigi daemon
    daemon = Daemon()
    daemon.run()