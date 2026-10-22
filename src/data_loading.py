
"""Compatibility helpers for loading the legacy WM-811K dataset."""

from pathlib import Path
import pickle
import zipfile


class LegacyPandasUnpickler(pickle.Unpickler):
    """Map old pandas module paths to their current locations."""

    def find_class(self, module, name):
        module = module.replace(
            "pandas.indexes",
            "pandas.core.indexes",
        )
        return super().find_class(module, name)


def load_wm811k(path):
    """Load LSWMD.pkl directly or from its original ZIP archive."""

    path = Path(path)

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as archive:
            pickle_files = [
                name
                for name in archive.namelist()
                if name.endswith(".pkl")
            ]

            if len(pickle_files) != 1:
                raise ValueError(
                    "The archive must contain exactly one pickle file."
                )

            with archive.open(pickle_files[0]) as file:
                return LegacyPandasUnpickler(
                    file,
                    encoding="latin1",
                ).load()

    with path.open("rb") as file:
        return LegacyPandasUnpickler(
            file,
            encoding="latin1",
        ).load()
