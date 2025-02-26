#!/usr/bin/env python3
import argparse
import logging
from franka_sim import FrankaSimServer

# Configure logging to silence Numba debug output
logging.getLogger('numba').setLevel(logging.WARNING)

def main():
    ## get command line arguments visualization
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    server = FrankaSimServer(enable_vis=args.vis)
    try:
        server.start()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.stop()

if __name__ == "__main__":
    main()
