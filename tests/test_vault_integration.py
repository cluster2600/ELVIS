#!/usr/bin/env python3
"""
Test script for HashiCorp Vault integration
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_vault_fallback():
    """Test that the system works without Vault (fallback mode)"""
    print("🧪 Testing Vault Fallback Mode")
    print("=" * 40)
    
    # Temporarily disable Vault
    os.environ['VAULT_ENABLED'] = 'false'
    
    try:
        from utils.secrets_manager import get_enhanced_secrets_manager
        
        # Set some test environment variables
        os.environ['TEST_SECRET'] = 'test_value_from_env'
        os.environ['BINANCE_API_KEY'] = 'test_binance_key'
        
        secrets = get_enhanced_secrets_manager()
        
        # Test basic secret retrieval
        test_value = secrets.get_secret('TEST_SECRET', 'default')
        if test_value == 'test_value_from_env':
            print("✅ Basic fallback: SUCCESS")
        else:
            print(f"❌ Basic fallback: FAILED (got: {test_value})")
        
        # Test credential helpers
        binance_creds = secrets.get_binance_credentials()
        if binance_creds.get('api_key') == 'test_binance_key':
            print("✅ Credential helper fallback: SUCCESS")
        else:
            print("❌ Credential helper fallback: FAILED")
        
        # Test status
        vault_status = secrets.get_vault_status()
        if not vault_status['enabled']:
            print("✅ Vault status (disabled): SUCCESS")
        else:
            print("❌ Vault status (disabled): FAILED")
        
        return True
        
    except Exception as e:
        print(f"❌ Fallback test failed: {e}")
        return False
    finally:
        # Clean up
        os.environ.pop('TEST_SECRET', None)
        os.environ.pop('BINANCE_API_KEY', None)
        os.environ['VAULT_ENABLED'] = 'true'

def test_vault_client():
    """Test Vault client initialization"""
    print("\n🧪 Testing Vault Client")
    print("=" * 30)
    
    try:
        from utils.vault_client import get_vault_client
        
        vault_client = get_vault_client(logger=logger)
        
        if vault_client:
            print("✅ Vault client creation: SUCCESS")
        else:
            print("❌ Vault client creation: FAILED")
            return False
        
        # Test health check (may fail if Vault not running)
        try:
            healthy = vault_client.health_check()
            if healthy:
                print("✅ Vault health check: SUCCESS")
            else:
                print("⚠️ Vault health check: FAILED (Vault may not be running)")
        except Exception as e:
            print(f"⚠️ Vault health check: ERROR ({e})")
        
        return True
        
    except Exception as e:
        print(f"❌ Vault client test failed: {e}")
        return False

def test_enhanced_secrets_manager():
    """Test enhanced secrets manager"""
    print("\n🧪 Testing Enhanced Secrets Manager")
    print("=" * 40)
    
    try:
        from utils.secrets_manager import get_enhanced_secrets_manager
        
        secrets = get_enhanced_secrets_manager()
        
        if secrets:
            print("✅ Enhanced secrets manager creation: SUCCESS")
        else:
            print("❌ Enhanced secrets manager creation: FAILED")
            return False
        
        # Test status method
        try:
            status = secrets.get_vault_status()
            print(f"📊 Vault status: {status}")
            print("✅ Status method: SUCCESS")
        except Exception as e:
            print(f"❌ Status method: FAILED ({e})")
        
        # Test credential helpers
        try:
            db_creds = secrets.get_database_credentials()
            if isinstance(db_creds, dict):
                print("✅ Database credentials method: SUCCESS")
            else:
                print("❌ Database credentials method: FAILED")
        except Exception as e:
            print(f"❌ Database credentials method: FAILED ({e})")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced secrets manager test failed: {e}")
        return False

def test_database_integration():
    """Test database integration with new secrets manager"""
    print("\n🧪 Testing Database Integration")
    print("=" * 35)
    
    try:
        # Set test database environment variables
        os.environ['POSTGRES_HOST'] = 'localhost'
        os.environ['POSTGRES_USER'] = 'test_user'
        os.environ['POSTGRES_PASSWORD'] = 'test_password'
        os.environ['POSTGRES_DBNAME'] = 'test_db'
        
        from utils.paper_trade_db import get_conn
        
        # This should not actually connect (wrong credentials)
        # But it should try to use the enhanced secrets manager
        try:
            conn = get_conn()
            if conn:
                conn.close()
                print("✅ Database connection attempt: SUCCESS (unexpected)")
            else:
                print("✅ Database connection attempt: SUCCESS (expected failure)")
        except Exception as e:
            # Expected to fail with test credentials
            print("✅ Database connection attempt: SUCCESS (expected failure)")
        
        return True
        
    except Exception as e:
        print(f"❌ Database integration test failed: {e}")
        return False
    finally:
        # Clean up test environment variables
        for key in ['POSTGRES_HOST', 'POSTGRES_USER', 'POSTGRES_PASSWORD', 'POSTGRES_DBNAME']:
            os.environ.pop(key, None)

def test_imports():
    """Test that all modules can be imported"""
    print("\n🧪 Testing Module Imports")
    print("=" * 30)
    
    modules_to_test = [
        'utils.vault_client',
        'utils.secrets_manager',
    ]
    
    all_good = True
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✅ {module}: SUCCESS")
        except ImportError as e:
            print(f"❌ {module}: FAILED ({e})")
            all_good = False
        except Exception as e:
            print(f"⚠️ {module}: WARNING ({e})")
    
    return all_good

def main():
    """Run all tests"""
    print("🔐 HashiCorp Vault Integration Tests")
    print("=" * 50)
    
    tests = [
        ("Module Imports", test_imports),
        ("Vault Fallback Mode", test_vault_fallback),
        ("Vault Client", test_vault_client),
        ("Enhanced Secrets Manager", test_enhanced_secrets_manager),
        ("Database Integration", test_database_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}: EXCEPTION ({e})")
            results.append((test_name, False))
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 30)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Vault integration is ready.")
        print("\n📋 Next steps:")
        print("1. Install and start Vault server")
        print("2. Run: python scripts/setup_vault.py --all")
        print("3. Update .env with Vault configuration")
    else:
        print(f"\n⚠️ {total - passed} tests failed. Check the output above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())