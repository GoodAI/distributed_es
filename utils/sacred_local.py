from badger_utils.sacred import SacredConfigFactory

SACRED_LOCAL = False  # global switch between local and shared Sacred storage


def get_sacred_storage():
    if SACRED_LOCAL:
        return SacredConfigFactory.local()
    else:
        return SacredConfigFactory.shared()
