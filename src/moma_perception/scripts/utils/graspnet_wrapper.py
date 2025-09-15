#!/home/igrape/miniconda3/envs/moma/bin/python3
import sys
import os

def import_graspnet_demo():
    """Safely import graspnet demo with path isolation"""
    
    # Save current sys.path
    original_path = sys.path.copy()
    
    try:
        # Add graspnet path at the beginning
        current_dir = os.path.dirname(os.path.abspath(__file__))
        graspnet_path = os.path.join(current_dir, 'graspnet-baseline')
        
        if graspnet_path not in sys.path:
            sys.path.insert(0, graspnet_path)
            
        # Also add subdirectories that demo.py requires
        models_path = os.path.join(graspnet_path, 'models')
        dataset_path = os.path.join(graspnet_path, 'dataset')
        utils_path = os.path.join(graspnet_path, 'utils')
        
        for path in [models_path, dataset_path, utils_path]:
            if path not in sys.path:
                sys.path.insert(0, path)
        
        # Change working directory temporarily
        original_cwd = os.getcwd()
        os.chdir(graspnet_path)
            
        # Import the demo module
        import demo
        return demo
        
    except ImportError as e:
        print(f"Failed to import graspnet demo: {e}")
        return None
    finally:
        # Restore original sys.path to avoid conflicts
        sys.path = original_path
        # Restore original working directory
        try:
            os.chdir(original_cwd)
        except:
            pass

# Global variable to hold the demo module
_graspnet_demo = None

def get_graspnet_demo():
    """Get graspnet demo module (singleton pattern)"""
    global _graspnet_demo
    if _graspnet_demo is None:
        _graspnet_demo = import_graspnet_demo()
    return _graspnet_demo

def run_graspnet_demo(color_img, depth_img, mask):
    """Run graspnet demo with given inputs"""
    demo = get_graspnet_demo()
    if demo is None:
        print("GraspNet demo not available")
        return None
        
    try:
        # Call the demo function
        result = demo.demo(color_img, depth_img, mask)
        return result
    except Exception as e:
        print(f"Error running graspnet demo: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        return None
