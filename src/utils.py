import ast
import os


def convert_config(parser):
    output = {}
    for key in parser:
        try:
            output[key] = int(parser[key])  # try if it is int
        except ValueError:
            try:
                output[key] = float(parser[key])  # try if it is float
            except ValueError:
                try:  # check if it is list
                    output[key] = ast.literal_eval(parser[key])
                except ValueError:
                    output[key] = parser[key]  # it is string
    return output


def data_check(data_path):
    try:
        if os.listdir(data_path) is not None:
            return True
        else:
            return False
    except FileNotFoundError:
        return False
