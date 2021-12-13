import argparse


parser = argparse.ArgumentParser('test')

parser.add_argument('config', help='hi')


args = parser.parse_args()

print(args)
