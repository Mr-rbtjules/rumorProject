import datetime as dt


def _to_ts(tstr):
    """ "Wed Jan 07 11:06:08 +0000 2015" to 1420628768"""
    fmt = "%a %b %d %H:%M:%S %z %Y"
    return int(dt.datetime.strptime(tstr, fmt).timestamp())


def testPackage() -> None:
    """ fct to test the package """
    print("Package is working")
    return None