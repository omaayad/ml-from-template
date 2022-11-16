import nox
from typing import List

PYTHON_VERSIONS: List[str] = ["3.9"]

nox.options.reuse_existing_virtualenvs = True
nox.options.error_on_missing_interpreters = True


@nox.session()
def tests(session):
    session.run("pytest", "tests")
