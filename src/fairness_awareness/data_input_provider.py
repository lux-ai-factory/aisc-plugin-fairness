import io

import pandas as pd
from aisc_plugin_interface import BaseInputProvider


class DataFrameProvider(BaseInputProvider):
    def _read_data(self, file_content: bytes) -> pd.DataFrame:
        import pandas as pd

        file_stream = io.BytesIO(file_content)
        try:
            return pd.read_parquet(file_stream)
        except Exception:
            file_stream.seek(0)
            try:
                return pd.read_csv(file_stream)
            except Exception as e:
                raise ValueError("File is neither a valid Parquet nor CSV.") from e
