# Meal Delivery Routing 2024

See the problem description for more details about the problem.

To get the code running, install the dependencies from environment.yml. The easiest way is to use the conda command:
`conda env create -f environment.yml`

You are provided with two relevant files of code: navigator and simulator. All methods you might need contain a docstring on how to use them. You can also run both files to get some examples on how to use them. At the bottom of both files, under `__name__ == "__main__"`, the examples can be found including some comments on what is happening.

In your actual code, you will probably use them like this:

```python
from navigator import Navigator
from simulator import Simulator

nav = Navigator("data/paris_map.txt")
sim = Simulator(year=2024, month=1, day=9)

...
```
