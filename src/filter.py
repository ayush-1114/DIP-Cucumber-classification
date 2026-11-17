import sys

for line in sys.stdin:
    # If the line does NOT contain Claude, print it.
    if "Co-Authored-By: Claude" not in line:
        sys.stdout.write(line)