import os
import subprocess

if __name__ == "__main__":

    # Assume the system is linux (convert it to match OS)

    # Check if folder exists, if not make it
    if not os.path.exists("venv/lib/python3.7/site-packages/scarf"):
        os.mkdir("venv/lib/python3.7/site-packages/scarf")
        subprocess.run("touch venv/lib/python3.7/site-packages/scarf/__init__.py")

    # Copy custom files
    subprocess.run("cp -f scarf/*.py venv/lib/python3.7/site-packages/scarf/")
    
