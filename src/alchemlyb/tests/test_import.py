import alchemlyb

def test_name():
    try:
        assert alchemlyb.__name__ == 'alchemlyb'
    except Exception as e:
        raise e
