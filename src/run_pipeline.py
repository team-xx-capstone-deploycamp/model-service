#!/usr/bin/env python3
import luigi
import os
from pipeline.tasks import MLPipeline

if __name__ == '__main__':
    # Set Luigi config file path
    os.environ['LUIGI_CONFIG_PATH'] = 'config/luigi.cfg'

    luigi.run(['MLPipeline', '--workers=1', '--local-scheduler'])