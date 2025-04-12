import pandas as pd
import asyncio
from datetime import datetime, timezone
from cybotrade_datasource import cybotrade_datasource
import numpy as np

API_KEY = "rJy3bDhtE1i87LLzrttZXanI4tOWfTPRmknoTFpRCcTsOf3U"

async def fetch_data(start_date=datetime(year=2024, month=1, day=1, tzinfo=timezone.utc),
                    end_date=datetime(year=2025, month=1, day=1, tzinfo=timezone.utc)):
    """
    Fetch data from CyboTrade API and perform initial cleaning
    """
    data = await cybotrade_datasource.query_paginated(
        api_key=API_KEY, 
        topic='cryptoquant|btc/inter-entity-flows/miner-to-miner?from_miner=f2pool&to_miner=all_miner&window=hour', 
        start_time=start_date,
        end_time=end_date
    )
    
    df = pd.DataFrame(data)
    
    # Parse datetime and set index
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.set_index('datetime')
    df = df.sort_index()
    
    # Convert object column to numeric
    df['transactions_count_flow'] = pd.to_numeric(df['transactions_count_flow'], errors='coerce')
    
    return df

if __name__ == "__main__":
    # Test the data loader independently
    async def test_loader():
        df = await fetch_data()
        print(df.head())
        print(df.shape)
        print(df.dtypes)
    
    asyncio.run(test_loader())