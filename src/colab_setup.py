"""
Google Colab setup utilities for xai-dark-matter-localization project.

This module handles:
- Mounting Google Drive in Colab
- Setting up the working directory
- Adding project paths to sys.path
- Environment configuration
"""

import os
import sys
from pathlib import Path


def is_colab():
    """Check if running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def setup_colab(project_folder="xai-dark-matter-localization", verbose=True):
    """
    Setup Google Colab environment for the project.
    
    Args:
        project_folder (str): Name of the project folder in Google Drive (MyDrive)
        verbose (bool): Print setup information
    
    Returns:
        dict: Information about the setup (working directory, data paths, etc.)
    """
    if not is_colab():
        if verbose:
            print("⚠️  Not running in Google Colab. Local setup only.")
        return {"env": "local", "is_colab": False}
    
    if verbose:
        print("🔧 Setting up Google Colab environment...")
    
    # Mount Google Drive
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        if verbose:
            print("✓ Google Drive mounted at /content/drive")
    except Exception as e:
        print(f"✗ Error mounting Google Drive: {e}")
        return {"env": "colab", "is_colab": True, "mount_success": False}
    
    # Change to project directory
    project_path = f"/content/drive/MyDrive/{project_folder}"
    try:
        os.chdir(project_path)
        if verbose:
            print(f"✓ Changed working directory to: {os.getcwd()}")
    except FileNotFoundError:
        print(f"✗ Project folder not found: {project_path}")
        return {"env": "colab", "is_colab": True, "mount_success": True, "chdir_success": False}
    
    # Add project to sys.path if not already there
    if project_path not in sys.path:
        sys.path.insert(0, project_path)
        if verbose:
            print(f"✓ Added project to sys.path")
    
    # Verify setup
    setup_info = {
        "env": "colab",
        "is_colab": True,
        "working_directory": os.getcwd(),
        "project_path": project_path,
        "python_version": sys.version.split()[0],
    }
    
    if verbose:
        print("\n📊 Setup Information:")
        print(f"  Working Directory: {setup_info['working_directory']}")
        print(f"  Project Path: {setup_info['project_path']}")
        
        # List main directories
        if os.path.exists("data"):
            print(f"  ✓ data/ directory found")
        if os.path.exists("src"):
            print(f"  ✓ src/ directory found")
        if os.path.exists("notebooks"):
            print(f"  ✓ notebooks/ directory found")
    
    return setup_info


def setup_colab_auto():
    """
    Auto-setup for Colab (call from config.py).
    Only runs in Colab environment, silent in local environment.
    """
    if is_colab():
        setup_colab(verbose=False)


# Auto-setup when imported from config.py
if is_colab():
    try:
        setup_colab(verbose=False)
    except Exception:
        pass  # Silently fail - user can call setup_colab() explicitly if needed
