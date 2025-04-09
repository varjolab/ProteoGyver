
import tomlkit
def read_toml(toml_file):
    with open(toml_file, 'r') as tf:
        data = tomlkit.load(tf)
    return data
