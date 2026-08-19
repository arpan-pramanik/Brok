import pandas as pd
import requests
from io import BytesIO

url = "https://huggingface.co/datasets/AI4Bharat/MSMARCO-XI/resolve/main/validation/asmval.parquet"
print(f"Downloading {url} ...")
resp = requests.get(url)
resp.raise_for_status()

df = pd.read_parquet(BytesIO(resp.content))
print("Columns:", df.columns)
print("First row:", df.iloc[0].to_dict())
