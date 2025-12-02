"""
Script to download the seismic dataset from Kaggle.
Requires kaggle API setup.
"""

import subprocess
import sys
import os

def setup_kaggle_api():
    """Instructions for setting up Kaggle API"""
    print("🔑 KAGGLE API SETUP REQUIRED")
    print("=" * 40)
    print("1. Go to https://www.kaggle.com/settings")
    print("2. Scroll to API section")
    print("3. Click 'Create New API Token'") 
    print("4. Download kaggle.json file")
    print("5. Place it in ~/.kaggle/ (Linux/Mac) or C:\\Users\\{username}\\.kaggle\\ (Windows)")
    print("6. Set permissions: chmod 600 ~/.kaggle/kaggle.json (Linux/Mac)")
    print("\nOr run: kaggle config path -p ~/.kaggle/kaggle.json")
    print("\nThen run this script again.")

def download_dataset():
    """Download the seismic dataset from Kaggle"""
    dataset_id = "ahmeduzaki/global-earthquake-tsunami-risk-assessment-dataset"
    
    try:
        print("🌍 Downloading seismic dataset from Kaggle...")
        print(f"Dataset: {dataset_id}")
        
        # Download using kaggle CLI
        result = subprocess.run([
            "kaggle", "datasets", "download", 
            dataset_id, "-p", ".", "--unzip"
        ], capture_output=True, text=True, check=True)
        
        print("✅ Dataset downloaded successfully!")
        
        # List downloaded files
        files = [f for f in os.listdir(".") if f.endswith('.csv')]
        if files:
            print(f"📁 CSV files found: {files}")
            
            # Rename to expected filename if needed
            main_csv = files[0]  # Assume first CSV is the main dataset
            if main_csv != "earthquake_tsunami_dataset.csv":
                os.rename(main_csv, "earthquake_tsunami_dataset.csv")
                print(f"📝 Renamed {main_csv} -> earthquake_tsunami_dataset.csv")
        
        print("\n🚀 Ready to run analysis!")
        print("Next: python run_analysis.py")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading dataset: {e}")
        print("\nPossible issues:")
        print("• Kaggle API not configured")
        print("• Internet connection")
        print("• Dataset permissions")
        print("\n💡 Try manual download:")
        print("1. Go to: https://www.kaggle.com/datasets/ahmeduzaki/global-earthquake-tsunami-risk-assessment-dataset")
        print("2. Click 'Download' button")
        print("3. Extract and place CSV as 'earthquake_tsunami_dataset.csv'")
        
    except FileNotFoundError:
        print("❌ Kaggle CLI not installed")
        print("Install with: pip install kaggle")
        print("Then configure API and try again")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--setup":
        setup_kaggle_api()
    else:
        download_dataset()

