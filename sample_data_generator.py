#!/usr/bin/env python3
"""
Sample Data Generator for Aave V2 Credit Scoring System

Creates realistic sample transaction data for testing the credit scoring system.
"""

import json
import random
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any

def generate_sample_data(num_wallets: int = 100, transactions_per_wallet: int = 50) -> List[Dict[str, Any]]:
    """
    Generate sample Aave V2 transaction data
    
    Args:
        num_wallets: Number of unique wallets to generate
        transactions_per_wallet: Average transactions per wallet
        
    Returns:
        List of transaction dictionaries
    """
    
    actions = ['deposit', 'borrow', 'repay', 'redeemunderlying', 'liquidationcall']
    reserves = [
        '0xa0b86a33e6afacad7a3bb6ce5b5b1e8e3e3e3e3e',  # USDC
        '0xb1c86a33e6afacad7a3bb6ce5b5b1e8e3e3e3e3e',  # USDT  
        '0xc2d86a33e6afacad7a3bb6ce5b5b1e8e3e3e3e3e',  # DAI
        '0xd3e86a33e6afacad7a3bb6ce5b5b1e8e3e3e3e3e',  # WETH
        '0xe4f86a33e6afacad7a3bb6ce5b5b1e8e3e3e3e3e',  # WBTC
        '0xf5086a33e6afacad7a3bb6ce5b5b1e8e3e3e3e3e',  # LINK
        '0x06186a33e6afacad7a3bb6ce5b5b1e8e3e3e3e3e',  # AAVE
        '0x17286a33e6afacad7a3bb6ce5b5b1e8e3e3e3e3e',  # UNI
    ]
    
    transactions = []
    start_date = datetime.now() - timedelta(days=365)
    
    for wallet_id in range(num_wallets):
        wallet_address = f"0x{wallet_id:040x}"
        
        # Define wallet behavior type
        behavior_type = random.choices([
            'conservative',   # Low risk, good patterns
            'moderate',       # Average behavior
            'aggressive',     # Higher risk behavior
            'problematic'     # Liquidations, erratic patterns
        ], weights=[0.3, 0.4, 0.2, 0.1])[0]
        
        # Generate transactions for this wallet
        wallet_transactions = random.randint(
            max(1, transactions_per_wallet - 30), 
            transactions_per_wallet + 30
        )
        
        wallet_start_date = start_date + timedelta(days=random.randint(0, 300))
        
        for tx_id in range(wallet_transactions):
            
            # Time progression
            tx_time = wallet_start_date + timedelta(
                days=np.random.exponential(5),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59),
                seconds=random.randint(0, 59)
            )
            
            # Choose action based on behavior type
            if behavior_type == 'conservative':
                action = random.choices(actions, weights=[0.4, 0.2, 0.25, 0.15, 0.0])[0]
                amount_range = (100, 10000)
            elif behavior_type == 'moderate':
                action = random.choices(actions, weights=[0.3, 0.25, 0.25, 0.18, 0.02])[0]
                amount_range = (50, 50000)
            elif behavior_type == 'aggressive':
                action = random.choices(actions, weights=[0.25, 0.35, 0.2, 0.15, 0.05])[0]
                amount_range = (10, 100000)
            else:  # problematic
                action = random.choices(actions, weights=[0.2, 0.4, 0.15, 0.15, 0.1])[0]
                amount_range = (1, 200000)
            
            # Generate amount
            if action == 'liquidationcall':
                amount = random.uniform(1000, 50000)
            else:
                amount = random.uniform(*amount_range)
            
            # Add some volatility for problematic wallets
            if behavior_type == 'problematic' and random.random() < 0.3:
                amount *= random.uniform(0.1, 10)
            
            # Choose reserve
            if behavior_type == 'conservative':
                # Prefer major stablecoins
                reserve = random.choices(reserves, weights=[0.3, 0.3, 0.2, 0.1, 0.05, 0.02, 0.02, 0.01])[0]
            else:
                reserve = random.choice(reserves)
            
            transaction = {
                'user': wallet_address,
                'action': action,
                'amount': round(amount, 6),
                'reserve': reserve,
                'timestamp': int(tx_time.timestamp()),
                'block_number': random.randint(15000000, 16000000),
                'transaction_hash': f"0x{''.join(random.choices('0123456789abcdef', k=64))}"
            }
            
            transactions.append(transaction)
    
    # Sort by timestamp
    transactions.sort(key=lambda x: x['timestamp'])
    
    print(f"Generated {len(transactions)} transactions for {num_wallets} wallets")
    print(f"Date range: {datetime.fromtimestamp(transactions[0]['timestamp'])} to {datetime.fromtimestamp(transactions[-1]['timestamp'])}")
    
    return transactions

def save_sample_data(filename: str = "sample_transactions.json", 
                    num_wallets: int = 100, 
                    transactions_per_wallet: int = 50) -> None:
    """Save sample data to JSON file"""
    data = generate_sample_data(num_wallets, transactions_per_wallet)
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Sample data saved to {filename}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate sample Aave V2 transaction data')
    parser.add_argument('--wallets', type=int, default=100, help='Number of wallets')
    parser.add_argument('--transactions', type=int, default=50, help='Average transactions per wallet')
    parser.add_argument('--output', type=str, default='sample_transactions.json', help='Output filename')
    
    args = parser.parse_args()
    
    save_sample_data(args.output, args.wallets, args.transactions)