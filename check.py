def parse_packages(file):
    pkgs = {}
    with open(file) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                parts = line.split()
                name = parts[0].lower()
                version = parts[1] if len(parts) > 1 else None
                pkgs[name] = version
    return pkgs

a = parse_packages("server1_conda_list.txt")
b = parse_packages("server2_conda_list.txt")

only_a = a.keys() - b.keys()
only_b = b.keys() - a.keys()
diff_versions = {k: (a[k], b[k]) for k in a.keys() & b.keys() if a[k] != b[k]}

print("Only in server1:", only_a)
print("Only in server2:", only_b)
print("Different versions:", diff_versions)