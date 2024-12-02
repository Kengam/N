import argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', default='../../output')
    args = parser.parse_args()
    return args