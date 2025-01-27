import os

def get_script_dir():
    try:
        # Works if we're running as a .py script
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # Fallback if __file__ is not defined (e.g. in Jupyter)
        return os.getcwd()