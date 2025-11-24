#!/usr/bin/env python3
"""
Simple compilation test for Intel XPU implementation.
This test verifies that the code can be imported without errors.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_import():
    """Test that the module can be imported."""
    try:
        import deep_ep_xpu
        print("✅ Successfully imported deep_ep_xpu module")
        return True
    except ImportError as e:
        print(f"❌ Failed to import deep_ep_xpu: {e}")
        print("\nNote: This is expected if the module hasn't been built yet.")
        print("To build the module, run:")
        print("  cd intel_xpu")
        print("  source /opt/intel/oneapi/setvars.sh")
        print("  ./build.sh")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_module_attributes():
    """Test that the module has expected attributes."""
    try:
        import deep_ep_xpu
        
        # Check for expected classes/functions
        expected_attrs = ['Buffer']
        
        for attr in expected_attrs:
            if hasattr(deep_ep_xpu, attr):
                print(f"✅ Found attribute: {attr}")
            else:
                print(f"❌ Missing attribute: {attr}")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Error checking attributes: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Intel XPU Compilation Test")
    print("=" * 60)
    print()
    
    # Test 1: Import
    print("Test 1: Module Import")
    print("-" * 60)
    import_success = test_import()
    print()
    
    if not import_success:
        print("Skipping remaining tests (module not available)")
        return 1
    
    # Test 2: Module attributes
    print("Test 2: Module Attributes")
    print("-" * 60)
    attr_success = test_module_attributes()
    print()
    
    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Import Test: {'✅ PASS' if import_success else '❌ FAIL'}")
    print(f"Attribute Test: {'✅ PASS' if attr_success else '❌ FAIL'}")
    print()
    
    if import_success and attr_success:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())

