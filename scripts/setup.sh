#!/bin/bash
# =============================================================================
# OpenRuntime Enhanced - Complete Setup Script
# Installs all dependencies and sets up the development/production environment
# Author: Nik Jois <nikjois@llamasearch.ai>
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PYTHON_VERSION="3.11"
RUST_VERSION="1.70"
NODE_VERSION="18"

# Functions
print_header() {
    echo -e "\n${BLUE}=============================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=============================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}[OK] $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[WARN] $1${NC}"
}

print_error() {
    echo -e "${RED}[FAIL] $1${NC}"
}

check_command() {
    if command -v "$1" >/dev/null 2>&1; then
        print_success "$1 is installed"
        return 0
    else
        print_warning "$1 is not installed"
        return 1
    fi
}

install_homebrew() {
    if ! command -v brew >/dev/null 2>&1; then
        print_header "Installing Homebrew"
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        
        # Add Homebrew to PATH for M1/M2 Macs
        if [[ $(uname -m) == "arm64" ]]; then
            echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
            eval "$(/opt/homebrew/bin/brew shellenv)"
        fi
        
        print_success "Homebrew installed"
    else
        print_success "Homebrew already installed"
    fi
}

install_python() {
    print_header "Setting up Python ${PYTHON_VERSION}"
    
    if command -v pyenv >/dev/null 2>&1; then
        print_success "pyenv already installed"
    else
        print_warning "Installing pyenv"
        brew install pyenv
        echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
        echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
        echo 'eval "$(pyenv init -)"' >> ~/.zshrc
        export PYENV_ROOT="$HOME/.pyenv"
        export PATH="$PYENV_ROOT/bin:$PATH"
        eval "$(pyenv init -)"
    fi
    
    # Install Python version if not available
    if ! pyenv versions | grep -q "${PYTHON_VERSION}"; then
        print_warning "Installing Python ${PYTHON_VERSION}"
        pyenv install ${PYTHON_VERSION}
    fi
    
    pyenv local ${PYTHON_VERSION}
    print_success "Python ${PYTHON_VERSION} configured"
}

install_rust() {
    print_header "Setting up Rust ${RUST_VERSION}"
    
    if command -v rustc >/dev/null 2>&1; then
        print_success "Rust already installed"
        rustup update
    else
        print_warning "Installing Rust"
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source ~/.cargo/env
    fi
    
    # Install specific version if needed
    rustup install ${RUST_VERSION}
    rustup default ${RUST_VERSION}
    
    # Install useful components
    rustup component add clippy rustfmt
    
    print_success "Rust ${RUST_VERSION} configured"
}

install_node() {
    print_header "Setting up Node.js ${NODE_VERSION}"
    
    if command -v node >/dev/null 2>&1; then
        print_success "Node.js already installed"
    else
        print_warning "Installing Node.js"
        brew install node@${NODE_VERSION}
        brew link node@${NODE_VERSION}
    fi
    
    print_success "Node.js configured"
}

install_system_dependencies() {
    print_header "Installing System Dependencies"
    
    # Essential system tools
    brew install \
        curl \
        wget \
        git \
        jq \
        htop \
        tree \
        postgresql@15 \
        redis \
        cmake \
        pkg-config \
        openssl \
        libffi
    
    print_success "System dependencies installed"
}

install_docker() {
    print_header "Setting up Docker"
    
    if command -v docker >/dev/null 2>&1; then
        print_success "Docker already installed"
    else
        print_warning "Installing Docker Desktop"
        brew install --cask docker
        print_warning "Please start Docker Desktop manually"
    fi
}

setup_python_environment() {
    print_header "Setting up Python Environment"
    
    # Upgrade pip and install essential tools
    python -m pip install --upgrade pip setuptools wheel
    
    # Install requirements
    if [ -f "requirements.txt" ]; then
        print_warning "Installing production dependencies"
        pip install -r requirements.txt
        print_success "Production dependencies installed"
    fi
    
    if [ -f "requirements-dev.txt" ]; then
        print_warning "Installing development dependencies"
        pip install -r requirements-dev.txt
        print_success "Development dependencies installed"
    fi
    
    # Install the package in development mode
    pip install -e .
    print_success "OpenRuntime Enhanced installed in development mode"
}

build_rust_components() {
    print_header "Building Rust Components"
    
    if [ -d "rust-openai-crate" ]; then
        cd rust-openai-crate
        
        # Build in release mode
        cargo build --release
        print_success "Rust components built"
        
        # Run tests
        cargo test
        print_success "Rust tests passed"
        
        cd ..
    else
        print_warning "Rust components directory not found, skipping"
    fi
}

setup_databases() {
    print_header "Setting up Databases"
    
    # Start PostgreSQL
    brew services start postgresql@15
    print_success "PostgreSQL started"
    
    # Create database if it doesn't exist
    if ! psql -lqt | cut -d \| -f 1 | grep -qw openruntime; then
        createdb openruntime
        print_success "Created openruntime database"
        
        # Initialize database schema
        if [ -f "scripts/init_db.sql" ]; then
            psql openruntime < scripts/init_db.sql
            print_success "Database schema initialized"
        fi
    else
        print_success "Database already exists"
    fi
    
    # Start Redis
    brew services start redis
    print_success "Redis started"
}

setup_pre_commit_hooks() {
    print_header "Setting up Pre-commit Hooks"
    
    if command -v pre-commit >/dev/null 2>&1; then
        pre-commit install
        print_success "Pre-commit hooks installed"
    else
        print_warning "pre-commit not found, skipping hooks setup"
    fi
}

create_config_files() {
    print_header "Creating Configuration Files"
    
    # Create config directory
    mkdir -p config
    
    # Create default configuration if it doesn't exist
    if [ ! -f "config/openruntime.yml" ]; then
        cat > config/openruntime.yml << EOF
# OpenRuntime Enhanced Configuration
# Author: Nik Jois <nikjois@llamasearch.ai>

server:
  host: "0.0.0.0"
  port: 8001
  reload: false
  log_level: "info"
  workers: 4

ai:
  openai_api_key: "\${OPENAI_API_KEY}"
  temperature: 0.7
  max_tokens: 2048
  max_concurrent_tasks: 20

gpu:
  fallback_to_cpu: true
  cache_max_size: 1000
  monitoring_enabled: true

security:
  jwt_secret: "\${JWT_SECRET}"
  enable_rate_limiting: true
  enable_threat_detection: true
  max_login_attempts: 5

monitoring:
  prometheus_enabled: true
  metrics_port: 9090
  log_metrics: true

database:
  url: "postgresql://postgres:password@localhost:5432/openruntime"
  
redis:
  url: "redis://localhost:6379"
EOF
        print_success "Created config/openruntime.yml"
    fi
    
    # Create environment file template
    if [ ! -f ".env.example" ]; then
        cat > .env.example << EOF
# OpenRuntime Enhanced Environment Variables
# Copy this file to .env and fill in your values

# Required: OpenAI API Key
OPENAI_API_KEY=sk-your-openai-api-key-here

# Optional: Other AI providers
ANTHROPIC_API_KEY=your-anthropic-key-here

# Security
JWT_SECRET=your-jwt-secret-here

# Database
DATABASE_URL=postgresql://postgres:password@localhost:5432/openruntime
REDIS_URL=redis://localhost:6379

# Application Settings
LOG_LEVEL=info
MAX_CONCURRENT_TASKS=20
CACHE_MAX_SIZE=1000
GPU_FALLBACK_TO_CPU=true

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_PASSWORD=admin123
EOF
        print_success "Created .env.example"
    fi
    
    # Create logs directory
    mkdir -p logs
    mkdir -p data
    mkdir -p cache
    mkdir -p temp
    
    print_success "Directory structure created"
}

run_tests() {
    print_header "Running Tests"
    
    # Run Python tests
    if command -v pytest >/dev/null 2>&1; then
        python -m pytest tests/ -v --tb=short
        print_success "Python tests completed"
    else
        print_warning "pytest not found, skipping Python tests"
    fi
    
    # Run Rust tests
    if [ -d "rust-openai-crate" ] && command -v cargo >/dev/null 2>&1; then
        cd rust-openai-crate
        cargo test
        print_success "Rust tests completed"
        cd ..
    fi
}

check_installation() {
    print_header "Verifying Installation"
    
    # Check Python installation
    if python -c "import openruntime_enhanced; print('OpenRuntime Enhanced imported successfully')" 2>/dev/null; then
        print_success "OpenRuntime Enhanced Python module working"
    else
        print_error "OpenRuntime Enhanced Python module failed to import"
    fi
    
    # Check CLI installation
    if command -v openruntime >/dev/null 2>&1; then
        print_success "OpenRuntime CLI installed"
        openruntime --help > /dev/null
        print_success "OpenRuntime CLI working"
    else
        print_error "OpenRuntime CLI not found"
    fi
    
    # Check database connection
    if psql -d openruntime -c "SELECT 1;" >/dev/null 2>&1; then
        print_success "Database connection working"
    else
        print_error "Database connection failed"
    fi
    
    # Check Redis connection
    if redis-cli ping >/dev/null 2>&1; then
        print_success "Redis connection working"
    else
        print_error "Redis connection failed"
    fi
}

print_next_steps() {
    print_header "Setup Complete! Next Steps:"
    
    echo -e "${GREEN}1. Configure your environment:${NC}"
    echo -e "   cp .env.example .env"
    echo -e "   # Edit .env with your API keys and settings"
    
    echo -e "\n${GREEN}2. Start the development server:${NC}"
    echo -e "   openruntime server --reload"
    
    echo -e "\n${GREEN}3. Access the application:${NC}"
    echo -e "   • API Documentation: http://localhost:8001/docs"
    echo -e "   • WebSocket: ws://localhost:8001/ws"
    echo -e "   • Health Check: http://localhost:8001/"
    
    echo -e "\n${GREEN}4. Try the CLI commands:${NC}"
    echo -e "   openruntime devices"
    echo -e "   openruntime status"
    echo -e "   openruntime benchmark --type comprehensive"
    
    echo -e "\n${GREEN}5. Start with Docker (alternative):${NC}"
    echo -e "   docker-compose up -d"
    
    echo -e "\n${GREEN}6. Run tests:${NC}"
    echo -e "   python -m pytest tests/ -v"
    
    echo -e "\n${YELLOW}Important Notes:${NC}"
    echo -e "   • Set OPENAI_API_KEY in your .env file for AI features"
    echo -e "   • GPU features require macOS with Metal support"
    echo -e "   • Check logs/ directory for application logs"
    echo -e "   • See README.md for detailed documentation"
    
    echo -e "\n${BLUE}Author: Nik Jois <nikjois@llamasearch.ai>${NC}"
}

# Main execution
main() {
    print_header "OpenRuntime Enhanced - Complete Setup"
    echo -e "${BLUE}Author: Nik Jois <nikjois@llamasearch.ai>${NC}\n"
    
    # Check if we're on macOS
    if [[ "$OSTYPE" != "darwin"* ]]; then
        print_warning "This script is optimized for macOS. Some features may not work on other platforms."
    fi
    
    # Install dependencies
    install_homebrew
    install_python
    install_rust
    install_node
    install_system_dependencies
    install_docker
    
    # Setup application
    setup_python_environment
    build_rust_components
    setup_databases
    setup_pre_commit_hooks
    create_config_files
    
    # Test installation
    run_tests
    check_installation
    
    # Show next steps
    print_next_steps
    
    print_success "OpenRuntime Enhanced setup completed successfully!"
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "OpenRuntime Enhanced Setup Script"
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --minimal      Install only essential dependencies"
        echo "  --docker-only  Setup for Docker development only"
        echo ""
        echo "Author: Nik Jois <nikjois@llamasearch.ai>"
        exit 0
        ;;
    --minimal)
        print_header "Minimal Setup Mode"
        install_homebrew
        install_python
        setup_python_environment
        create_config_files
        print_success "Minimal setup completed!"
        ;;
    --docker-only)
        print_header "Docker-Only Setup Mode"
        install_homebrew
        install_docker
        create_config_files
        print_success "Docker setup completed!"
        echo "Run: docker-compose up -d"
        ;;
    *)
        main
        ;;
esac 