import pandas as pd

try:
    df = pd.read_csv("/home/debian/pgvector/pgvectorscale-rag-solution/data/wip.csv", sep=",", encoding="utf-8", lineterminator='\n')
    print("Column names:")
    for col in df.columns:
        print(f"'{col}'")
    print("Special Notes column:", df["Special Notes"])
except Exception as e:
    print(f"Error: {e}")
