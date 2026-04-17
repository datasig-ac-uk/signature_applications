# signature_applications

## Development

This repo uses `pre-commit`, which will automatically format your code and run some basic checks before you commit:

```
pip install pre-commit  # or brew install pre-commit on macOS
pre-commit install  # will install a pre-commit hook into the git repo
```

After doing this, each time you commit, some linters will be applied to format
the codebase. You can also/alternatively run `pre-commit run --all-files` to run
the checks.

### Warning

This notebooks in this repository are for demonstration purposes only. The code here might not be suitable for
deployment. Some of the dependencies are out of date and contain known security vulnerabilities. We strongly recommend
that you review the code and dependencies for each notebook to ensure they are consistent with your organization's
security policies, current best-practices, and any compliance requirements before using them for any production
deployments.


