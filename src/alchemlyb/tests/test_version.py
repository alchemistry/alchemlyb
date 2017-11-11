import alchemlyb

def test_version():
    try:
        version = alchemlyb.__version__
    except AttributeError:
        raise AssertionError("alchemlyb does not have __version__")

    assert len(version) > 0

def test_version_get_versions():
    import alchemlyb._version
    version = alchemlyb._version.get_versions()

    assert alchemlyb.__version__ == version["version"]

