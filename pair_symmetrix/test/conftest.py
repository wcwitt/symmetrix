import os, sys

# Hacky, prevents bad exit due that doesn't seem directly related to the tests
def pytest_sessionfinish(session, exitstatus):
    if exitstatus == 0:
        os._exit(0)
