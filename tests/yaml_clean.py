import argparse

import yaml

from axelera.app.pipe import gst

parser = argparse.ArgumentParser(description='Clean YAML file')
parser.add_argument('input', type=str, help='input YAML file')
parser.add_argument(
    'output', type=str, nargs='?', help='output YAML file (default: overwrite input)'
)


def yaml_clean(s):
    got = yaml.load(s, Loader=yaml.FullLoader)
    got = gst._add_element_names(got)
    return yaml.dump(got, sort_keys=False)


if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.input, 'r') as f:
        s = f.read()
    s = yaml_clean(s)
    with open(args.output or args.input, 'w') as f:
        f.write(s)
