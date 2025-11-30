## Package with loss function implementations

To add your own loss function and use it, simply:

1. Create a new file with the implementation
2. Add it to the package - import the new loss function in `__init__.py`

Now you can specify the new loss function and its arguments in the experiment configuration:

```yaml
criterion:
name: MyCustomLoss
args:
some_arg1: value1
some_arg2: value2
...
```