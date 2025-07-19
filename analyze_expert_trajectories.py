#!/usr/bin/env python3
"""Analyze expert trajectories to understand the data structure and dimensions."""

import pickle

import numpy as np


def analyze_expert_trajectories():
    """Load and analyze expert trajectories."""
    try:
        with open('/Users/kazuki-h/rl/kazoo/data/expert_trajectories.pkl', 'rb') as f:
            data = pickle.load(f)
        
        print("Expert Trajectories Analysis:")
        print("=" * 50)
        print(f"Data type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"Number of keys: {len(data.keys())}")
            print(f"Keys: {list(data.keys())[:10]}...")  # Show first 10 keys
            
            # Analyze first trajectory
            if data:
                first_key = list(data.keys())[0]
                first_traj = data[first_key]
                print(f"\nFirst trajectory (key: {first_key}):")
                print(f"Type: {type(first_traj)}")
                if hasattr(first_traj, '__len__'):
                    print(f"Length: {len(first_traj)}")
                
                # If it's a list/tuple, examine first element
                if isinstance(first_traj, (list, tuple)) and len(first_traj) > 0:
                    first_elem = first_traj[0]
                    print(f"First element type: {type(first_elem)}")
                    if hasattr(first_elem, 'shape'):
                        print(f"First element shape: {first_elem.shape}")
                    elif hasattr(first_elem, '__len__'):
                        print(f"First element length: {len(first_elem)}")
        
        elif isinstance(data, (list, tuple)):
            print(f"Data length: {len(data)}")
            if len(data) > 0:
                print(f"First element type: {type(data[0])}")
                if hasattr(data[0], 'shape'):
                    print(f"First element shape: {data[0].shape}")
        
        return data
        
    except Exception as e:
        print(f"Error loading expert trajectories: {e}")
        return None

if __name__ == "__main__":
    analyze_expert_trajectories()
