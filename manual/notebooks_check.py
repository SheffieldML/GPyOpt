import os
import sys
import subprocess
import tempfile

import nbformat

def check_notebooks_for_errors(notebooks_directory):
    ''' Evaluates all notebooks in given directory and prints errors, if any '''
    print("Checking notebooks in directory {} for errors".format(notebooks_directory))

    failed_notebooks_count = 0
    for file in os.listdir(notebooks_directory):
        if file.endswith(".ipynb"):
            print("Checking notebook " + file)
            full_file_path = os.path.join(notebooks_directory, file)
            output, errors = run_notebook(full_file_path)
            if errors is not None and len(errors) > 0:
                failed_notebooks_count += 1
                print("Errors in notebook " + file)
                print(errors)

    if failed_notebooks_count == 0:
        print("No errors found in notebooks under " + notebooks_directory)

def run_notebook(notebook_path):
    """Execute a notebook via nbconvert and collect output.
       :returns (parsed nb object, execution errors)
    """
    dirname, __ = os.path.split(notebook_path)
    os.chdir(dirname)
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        args = ["jupyter-nbconvert", "--to", "notebook", "--execute", "--allow-errors",
          "--ExecutePreprocessor.timeout=300",
          "--output", fout.name, notebook_path]
        try:
            subprocess.check_call(args)
        except subprocess.CalledProcessError as e:
            if e.returncode == 1:
              # print the message and ignore error with code 1 as this indicates there were errors in the notebook
              print(e.output)
              pass
            else:
              # all other codes indicate some other problem, rethrow
              raise

        fout.seek(0)
        nb = nbformat.read(fout, nbformat.current_nbformat)

    errors = [output for cell in nb.cells if "outputs" in cell
                     for output in cell["outputs"]\
                     if output.output_type == "error"]

    return nb, errors

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        notebooks_directory = os.getcwd()
    else:
        notebooks_directory = sys.argv[1]

    check_notebooks_for_errors(notebooks_directory)