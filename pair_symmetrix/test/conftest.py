from lammps import lammps


def pytest_sessionfinish(session, exitstatus):
    lammps(cmdargs=["-screen", "none", "-log", "none"]).finalize()
