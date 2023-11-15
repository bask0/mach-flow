import yaml
import tempfile
import os

class BaseSearchSpace(object):
    def __init__(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            self.hp_path = f.name

        self.config = None

    def dump_condig(self):

        if not hasattr(self, 'config'):
            raise RuntimeError(
                'this instance has not attribute \'condig\'. Most likely, the parent class has not been '
                'called in the subclass initialization.'
            )
        elif self.config is None:
            raise RuntimeError(
                'this instance`s \'config\' attribute is None. Most likely, the config has not been '
                'set in the subclass initialization.'
            )

        with open(self.hp_path, 'w') as file:
            yaml.dump(self.config, file)

    def teardown(self) -> None:
        os.remove(self.hp_path)


class FakeTrial(object):
    _trial_id: int = 0
