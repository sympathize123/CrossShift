#!/usr/bin/env python3
"""
Script to identify which specific user is causing the POI clustering error.
This will test each user's location data and find the problematic one.
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
from poi import PoiCluster

# Paths (repo-local)
DATASET_DIR = Path(__file__).resolve().parents[1]
RAW_ROOT = DATASET_DIR.parents[2] / "data" / "raw" / "D-2"
PATH_DATA = str(RAW_ROOT)
PATH_SENSOR = os.path.join(PATH_DATA, "newdata")

def test_user_location_data(user_id):
    """Test a specific user's location data for POI clustering issues."""
    location_file = os.path.join(PATH_SENSOR, user_id, 'Location.csv')
    
    if not os.path.exists(location_file):
        print(f"❌ {user_id}: Location.csv not found")
        return False
    
    try:
        # Read location data
        data = pd.read_csv(location_file)
        print(f"📊 {user_id}: {len(data)} total location points")
        
        # Filter by accuracy
        filtered_data = data[data['accuracy'] < 100]
        print(f"📊 {user_id}: {len(filtered_data)} points after accuracy filter")
        
        if len(filtered_data) < 3:
            print(f"❌ {user_id}: Insufficient data points ({len(filtered_data)})")
            return False
        
        # Convert to radians for clustering
        latlon_rad = np.radians(filtered_data[['latitude', 'longitude']].to_numpy())
        timestamps = filtered_data['timestamp'].values
        
        # Test with current parameters
        try:
            cluster = PoiCluster(
                d_max=100,  # Current parameter
                r_max=250,  # Current parameter
                t_max=60 * 60 * 1000,  # Current parameter
                t_min=5 * 60 * 1000    # Current parameter
            ).fit(X=latlon_rad, timestamps=timestamps)
            
            labels = cluster.predict(X=latlon_rad)
            unique_clusters = np.unique(labels)
            print(f"✅ {user_id}: POI clustering successful, {len(unique_clusters)} clusters found")
            return True
            
        except ValueError as e:
            print(f"❌ {user_id}: POI clustering failed - {e}")
            
            # Test with more lenient parameters
            try:
                cluster = PoiCluster(
                    d_max=200,  # More lenient
                    r_max=500,  # More lenient
                    t_max=120 * 60 * 1000,  # More lenient
                    t_min=2 * 60 * 1000     # More lenient
                ).fit(X=latlon_rad, timestamps=timestamps)
                
                labels = cluster.predict(X=latlon_rad)
                unique_clusters = np.unique(labels)
                print(f"⚠️  {user_id}: Works with lenient parameters, {len(unique_clusters)} clusters")
                return False
                
            except ValueError as e2:
                print(f"❌ {user_id}: Still fails even with lenient parameters - {e2}")
                return False
                
    except Exception as e:
        print(f"❌ {user_id}: Error reading data - {e}")
        return False

def main():
    """Test all users and identify problematic ones."""
    print("🔍 Testing all users for POI clustering issues...")
    print("=" * 60)
    
    # Get all user directories
    user_dirs = [d for d in os.listdir(PATH_SENSOR) if d.startswith('P')]
    user_dirs.sort()
    
    problematic_users = []
    working_users = []
    
    for user_id in user_dirs:
        print(f"\n--- Testing {user_id} ---")
        if test_user_location_data(user_id):
            working_users.append(user_id)
        else:
            problematic_users.append(user_id)
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"Total users tested: {len(user_dirs)}")
    print(f"Working users: {len(working_users)}")
    print(f"Problematic users: {len(problematic_users)}")
    
    if problematic_users:
        print(f"\n❌ PROBLEMATIC USERS:")
        for user in problematic_users:
            print(f"  - {user}")
    
    if working_users:
        print(f"\n✅ WORKING USERS:")
        for user in working_users[:10]:  # Show first 10
            print(f"  - {user}")
        if len(working_users) > 10:
            print(f"  ... and {len(working_users) - 10} more")

if __name__ == "__main__":
    main() 
