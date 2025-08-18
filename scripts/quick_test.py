"""
Quick test without dependencies to verify file structure and basic imports
"""

import os
import sys
from pathlib import Path

def test_file_structure():
    """Test that all required files exist"""
    print("🔍 Testing file structure...")
    
    base_path = Path(__file__).parent.parent
    
    required_files = [
        "main.py",
        "requirements.txt",
        "Dockerfile",
        "docker-compose.yml",
        ".env.example",
        "pytest.ini",
        "config/__init__.py",
        "config/settings.py",
        "agents/__init__.py",
        "agents/orchestrator.py",
        "agents/visual_search_agent.py",
        "agents/recommendation_agent.py",
        "agents/inventory_agent.py",
        "models/__init__.py",
        "models/schemas.py",
        "utils/__init__.py",
        "utils/monitoring.py",
        "utils/logging_config.py",
        "tests/__init__.py",
        "tests/conftest.py",
        "tests/test_main.py",
        "tests/test_agents.py",
        "tests/test_models.py",
        "scripts/__init__.py",
        "scripts/setup_firestore.py",
    ]
    
    missing_files = []
    existing_files = []
    
    for file_path in required_files:
        full_path = base_path / file_path
        if full_path.exists():
            existing_files.append(file_path)
        else:
            missing_files.append(file_path)
    
    print(f"✓ Found {len(existing_files)} required files")
    
    if missing_files:
        print(f"✗ Missing {len(missing_files)} files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    
    print("✓ All required files exist")
    return True

def test_python_syntax():
    """Test that Python files have valid syntax"""
    print("\n🐍 Testing Python syntax...")
    
    base_path = Path(__file__).parent.parent
    
    python_files = []
    for root, dirs, files in os.walk(base_path):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                python_files.append(file_path)
    
    syntax_errors = []
    valid_files = []
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            # Compile to check syntax
            compile(source, str(file_path), 'exec')
            valid_files.append(file_path)
            
        except SyntaxError as e:
            syntax_errors.append((file_path, str(e)))
        except Exception as e:
            syntax_errors.append((file_path, f"Error reading file: {str(e)}"))
    
    print(f"✓ {len(valid_files)} Python files have valid syntax")
    
    if syntax_errors:
        print(f"✗ {len(syntax_errors)} files have syntax errors:")
        for file_path, error in syntax_errors:
            print(f"  - {file_path}: {error}")
        return False
    
    return True

def test_config_structure():
    """Test configuration file structure"""
    print("\n⚙️  Testing configuration...")
    
    base_path = Path(__file__).parent.parent
    
    # Check .env.example
    env_example = base_path / ".env.example"
    if env_example.exists():
        with open(env_example, 'r') as f:
            content = f.read()
            
        required_vars = [
            "GOOGLE_CLOUD_PROJECT",
            "GOOGLE_APPLICATION_CREDENTIALS", 
            "VERTEX_AI_MODEL",
            "API_KEY",
            "FIRESTORE_DATABASE"
        ]
        
        missing_vars = []
        for var in required_vars:
            if var not in content:
                missing_vars.append(var)
        
        if missing_vars:
            print(f"✗ Missing environment variables in .env.example: {missing_vars}")
            return False
        else:
            print("✓ .env.example contains all required variables")
    else:
        print("✗ .env.example not found")
        return False
    
    # Check requirements.txt
    requirements = base_path / "requirements.txt"
    if requirements.exists():
        with open(requirements, 'r') as f:
            content = f.read()
            
        required_packages = [
            "fastapi",
            "uvicorn",
            "pydantic",
            "google-cloud-aiplatform",
            "crewai"
        ]
        
        missing_packages = []
        for package in required_packages:
            if package not in content:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"✗ Missing packages in requirements.txt: {missing_packages}")
            return False
        else:
            print("✓ requirements.txt contains key packages")
    else:
        print("✗ requirements.txt not found")
        return False
    
    return True

def test_docker_config():
    """Test Docker configuration"""
    print("\n🐳 Testing Docker configuration...")
    
    base_path = Path(__file__).parent.parent
    
    # Check Dockerfile
    dockerfile = base_path / "Dockerfile"
    if dockerfile.exists():
        with open(dockerfile, 'r') as f:
            content = f.read()
        
        if "FROM python" in content and "COPY requirements.txt" in content:
            print("✓ Dockerfile looks valid")
        else:
            print("✗ Dockerfile structure seems incorrect")
            return False
    else:
        print("✗ Dockerfile not found")
        return False
    
    # Check docker-compose.yml
    compose_file = base_path / "docker-compose.yml"
    if compose_file.exists():
        with open(compose_file, 'r') as f:
            content = f.read()
        
        if "ai-agent:" in content and "redis:" in content:
            print("✓ docker-compose.yml looks valid")
        else:
            print("✗ docker-compose.yml structure seems incorrect")
            return False
    else:
        print("✗ docker-compose.yml not found")
        return False
    
    return True

def main():
    """Run all quick tests"""
    print("🧪 E-Commerce AI Agent - Quick Structure Test")
    print("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Python Syntax", test_python_syntax),
        ("Configuration", test_config_structure),
        ("Docker Setup", test_docker_config),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
    
    print("\n" + "=" * 60)
    print("📊 QUICK TEST SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}/{total} tests ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 Project structure looks good!")
        print("\n📋 Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Configure environment: cp .env.example .env")
        print("3. Set up Google Cloud credentials")
        print("4. Run application: python main.py")
        print("5. Run full tests: pytest")
    else:
        print(f"\n⚠️  {total - passed} issues found. Please fix them before proceeding.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)