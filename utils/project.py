import os
import git

repo = git.Repo(".", search_parent_directories=True)
rootdir = repo.git.rev_parse("--show-toplevel")


def template(root, ignore=['.git']):
    """ Given a root directory, recursively iterate over all project directories 
        and generate a list of found directories. Then, generate a script to 
        recreate project directories.
    """
    folders = []
    for dirpath, dirnames, _ in os.walk(root):

        # 
        # add folder paths, excluding __pycache__ directories
        # 
        dirnames[:] = [dir for dir in dirnames if dir != '__pycache__' and dir not in ignore]
        for dirname in dirnames:
            folders.append(os.path.join(dirpath, dirname))


    code = "import os\n\n"
    for folder in folders:
        relative_path = os.path.relpath(folder)
        code += f"os.makedirs(r'{relative_path}', exist_ok=True)\n"

    script = os.path.join(rootdir, 'utils', 'template.py')
    with open(script, 'w') as file:
        file.write(code)
