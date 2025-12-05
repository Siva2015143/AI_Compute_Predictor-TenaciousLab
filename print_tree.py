import os

def print_tree(startpath, prefix=""):
    items = sorted(os.listdir(startpath))
    for i, item in enumerate(items):
        path = os.path.join(startpath, item)
        connector = "└── " if i == len(items) - 1 else "├── "
        print(prefix + connector + item)
        
        # Check if it's a directory we should skip traversing into
        if os.path.isdir(path) and item in [".venv", "venv"]:
            continue  # Skip traversing into these directories
        elif os.path.isdir(path):
            extension = "    " if i == len(items) - 1 else "│   "
            print_tree(path, prefix + extension)

if __name__ == "__main__":
    root_dir = "."  # current folder
    print(os.path.basename(os.path.abspath(root_dir)) + "/")
    print_tree(root_dir)