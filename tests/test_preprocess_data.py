# tests/test_preprocess_data.py

import unittest
import pandas as pd
from data.preprocess_data import preprocess_data

class TestPreprocessData(unittest.TestCase):
    def test_preprocess_returns(self):
        # Mock price_data with synthetic data
        price_data = pd.DataFrame({
            'AAPL': [150, 152, 153, 155, 157],
            'MSFT': [250, 252, 251, 255, 260],
            'GOOGL': [2700, 2710, 2720, 2730, 2740],
            'AMZN': [3300, 3320, 3310, 3330, 3350],
            'META': [350, 352, 351, 353, 355],
            'JPM': [160, 162, 161, 163, 165],
            'XOM': [60, 61, 62, 63, 64],
            'JNJ': [140, 142, 141, 143, 145],
            'VZ': [55, 56, 55, 57, 58],
            'PG': [130, 132, 131, 133, 135]
        })
        returns, features = preprocess_data(price_data)
        
        # Check if returns are correctly calculated
        expected_returns = price_data.pct_change().dropna()
        pd.testing.assert_frame_equal(returns, expected_returns)
        
        # Check if features are calculated without NaNs
        self.assertFalse(features.isnull().values.any())
        
        # Check if the index aligns
        self.assertListEqual(list(returns.index), list(features.index))
        
        # Check the number of features
        expected_num_features = len(price_data.columns) * 6  # SMA_5, SMA_10, RSI_14, MACD, MACD_Signal, ATR_14
        self.assertEqual(features.shape[1], expected_num_features)

if __name__ == '__main__':
    unittest.main()