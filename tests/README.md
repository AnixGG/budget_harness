# tests

Run from the repository root with the locked dependencies installed:

```bash
python -m unittest discover -s tests -v
```

Table arithmetic/typing tests, direct action tests and release runner integration
use temporary files and controlled provider responses. No API key or network is used.
