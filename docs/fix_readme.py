with open('../README.md', 'r') as file:
    content = file.read()
content = content.replace(
    '_images/', 'https://github.com/UN-GCPDS/RS/raw/master/docs/source/notebooks/_images/')
with open('../README.md', 'w') as file:
    file.write(content)
