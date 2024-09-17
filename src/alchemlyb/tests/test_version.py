import alchemlyb


def test_version():
    try:
        version = alchemlyb.__version__
    except AttributeError:
        raise AssertionError("alchemlyb does not have __version__")

    assert len(version) > 0


def test_version__version__():
    import alchemlyb._version

    assert alchemlyb.__version__ == alchemlyb._version.__version__
