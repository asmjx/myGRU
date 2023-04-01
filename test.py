import models
models.__dict__
for item in models.__dict__:
    print(item,models.__dict__[item])
...