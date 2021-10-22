import Orange.data
import os.path


# a no-op workaround so that table unlocking does not crash with older Orange
# remove when the minimum supported version is 3.31
from Orange.data import Table
from contextlib import nullcontext
if not hasattr(Table, "unlocked"):
    Table.unlocked = nullcontext
    Table.unlocked_reference = nullcontext


from . import io  # register file formats


def get_sample_datasets_dir():
    thispath = os.path.dirname(__file__)
    dataset_dir = os.path.join(thispath, 'datasets')
    return os.path.realpath(dataset_dir)


Orange.data.table.dataset_dirs.append(get_sample_datasets_dir())
