#!/bin/bash

# Open Memory Suite Deployment Script
# Provides one-command deployment for different configurations

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="open-memory-suite"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Open Memory Suite Deployment Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    dev         Deploy development environment
    prod        Deploy production environment  
    gpu         Deploy with GPU acceleration
    stop        Stop all services
    restart     Restart services
    logs        Show service logs
    status      Show service status
    clean       Clean up containers and volumes
    backup      Backup data and models
    restore     Restore from backup

Options:
    --build     Force rebuild of containers
    --gpu       Enable GPU support (for prod command)
    --scale N   Scale to N instances
    --help      Show this help message

Examples:
    $0 dev                    # Start development environment
    $0 prod --build           # Build and start production environment
    $0 gpu                    # Start with GPU acceleration
    $0 logs open-memory-suite # Show application logs
    $0 clean                  # Clean up everything

EOF
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running. Please start Docker first."
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Create necessary directories
create_directories() {
    log_info "Creating necessary directories..."
    
    mkdir -p "$SCRIPT_DIR/data"
    mkdir -p "$SCRIPT_DIR/logs"
    mkdir -p "$SCRIPT_DIR/models"
    mkdir -p "$SCRIPT_DIR/storage"
    mkdir -p "$SCRIPT_DIR/monitoring/prometheus"
    mkdir -p "$SCRIPT_DIR/monitoring/grafana/dashboards"
    mkdir -p "$SCRIPT_DIR/monitoring/grafana/datasources"
    
    log_success "Directories created"
}

# Deploy development environment
deploy_dev() {
    log_info "Deploying development environment..."
    
    create_directories
    
    # Create development override file
    cat > "$SCRIPT_DIR/docker-compose.override.yml" << EOF
version: '3.8'
services:
  open-memory-suite:
    build:
      target: development
    environment:
      - ENVIRONMENT=development
      - LOG_LEVEL=debug
      - RELOAD=true
    volumes:
      - .:/app
    command: ["python", "-m", "open_memory_suite.server", "--reload", "--debug"]
EOF
    
    docker-compose up -d ${BUILD_FLAG}
    
    log_success "Development environment deployed!"
    log_info "Access the application at: http://localhost:8000"
    log_info "Access monitoring at: http://localhost:3000 (admin/admin)"
}

# Deploy production environment
deploy_prod() {
    log_info "Deploying production environment..."
    
    create_directories
    
    # Remove override file if exists
    rm -f "$SCRIPT_DIR/docker-compose.override.yml"
    
    # Enable GPU if requested
    if [ "$GPU_ENABLED" = "true" ]; then
        log_info "Enabling GPU acceleration..."
        cat > "$SCRIPT_DIR/docker-compose.override.yml" << EOF
version: '3.8'
services:
  open-memory-suite:
    build:
      target: gpu
    environment:
      - GPU_ENABLED=true
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
EOF
    fi
    
    docker-compose up -d ${BUILD_FLAG}
    
    log_success "Production environment deployed!"
    log_info "Access the application at: http://localhost:8000"
    log_info "Access web interface at: http://localhost:8501"
    log_info "Access monitoring at: http://localhost:3000"
}

# Deploy GPU environment
deploy_gpu() {
    log_info "Deploying GPU-accelerated environment..."
    
    # Check for NVIDIA Docker
    if ! docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi &> /dev/null; then
        log_error "NVIDIA Docker runtime not available. Please install nvidia-docker2."
        exit 1
    fi
    
    GPU_ENABLED=true deploy_prod
    
    log_success "GPU-accelerated environment deployed!"
}

# Stop services
stop_services() {
    log_info "Stopping services..."
    docker-compose down
    log_success "Services stopped"
}

# Restart services
restart_services() {
    log_info "Restarting services..."
    docker-compose restart
    log_success "Services restarted"
}

# Show logs
show_logs() {
    if [ -n "$2" ]; then
        docker-compose logs -f "$2"
    else
        docker-compose logs -f
    fi
}

# Show status
show_status() {
    log_info "Service status:"
    docker-compose ps
    
    echo
    log_info "Resource usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"
}

# Clean up
clean_up() {
    log_warning "This will remove all containers, networks, and volumes. Are you sure? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        log_info "Cleaning up..."
        docker-compose down -v --remove-orphans
        docker system prune -f
        log_success "Cleanup completed"
    else
        log_info "Cleanup cancelled"
    fi
}

# Backup data
backup_data() {
    BACKUP_DIR="$SCRIPT_DIR/backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    log_info "Creating backup in $BACKUP_DIR..."
    
    # Backup application data
    if [ -d "$SCRIPT_DIR/data" ]; then
        cp -r "$SCRIPT_DIR/data" "$BACKUP_DIR/"
    fi
    
    if [ -d "$SCRIPT_DIR/models" ]; then
        cp -r "$SCRIPT_DIR/models" "$BACKUP_DIR/"
    fi
    
    if [ -d "$SCRIPT_DIR/storage" ]; then
        cp -r "$SCRIPT_DIR/storage" "$BACKUP_DIR/"
    fi
    
    # Backup database
    if docker-compose ps | grep -q postgres; then
        log_info "Backing up database..."
        docker-compose exec -T postgres pg_dump -U memory_user memory_suite > "$BACKUP_DIR/database.sql"
    fi
    
    # Create archive
    cd "$SCRIPT_DIR/backups"
    tar -czf "$(basename "$BACKUP_DIR").tar.gz" "$(basename "$BACKUP_DIR")"
    rm -rf "$BACKUP_DIR"
    
    log_success "Backup created: $SCRIPT_DIR/backups/$(basename "$BACKUP_DIR").tar.gz"
}

# Restore from backup
restore_data() {
    if [ -z "$2" ]; then
        log_error "Please specify backup file: $0 restore <backup-file>"
        exit 1
    fi
    
    BACKUP_FILE="$2"
    
    if [ ! -f "$BACKUP_FILE" ]; then
        log_error "Backup file not found: $BACKUP_FILE"
        exit 1
    fi
    
    log_warning "This will overwrite existing data. Are you sure? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        log_info "Restoring from backup..."
        
        # Extract backup
        TEMP_DIR=$(mktemp -d)
        tar -xzf "$BACKUP_FILE" -C "$TEMP_DIR"
        BACKUP_DIR=$(find "$TEMP_DIR" -maxdepth 1 -type d | tail -1)
        
        # Restore data
        if [ -d "$BACKUP_DIR/data" ]; then
            rm -rf "$SCRIPT_DIR/data"
            cp -r "$BACKUP_DIR/data" "$SCRIPT_DIR/"
        fi
        
        if [ -d "$BACKUP_DIR/models" ]; then
            rm -rf "$SCRIPT_DIR/models"
            cp -r "$BACKUP_DIR/models" "$SCRIPT_DIR/"
        fi
        
        if [ -d "$BACKUP_DIR/storage" ]; then
            rm -rf "$SCRIPT_DIR/storage"
            cp -r "$BACKUP_DIR/storage" "$SCRIPT_DIR/"
        fi
        
        # Restore database
        if [ -f "$BACKUP_DIR/database.sql" ] && docker-compose ps | grep -q postgres; then
            log_info "Restoring database..."
            docker-compose exec -T postgres psql -U memory_user -d memory_suite < "$BACKUP_DIR/database.sql"
        fi
        
        rm -rf "$TEMP_DIR"
        log_success "Restore completed"
    else
        log_info "Restore cancelled"
    fi
}

# Parse command line arguments
BUILD_FLAG=""
GPU_ENABLED="false"
SCALE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --build)
            BUILD_FLAG="--build"
            shift
            ;;
        --gpu)
            GPU_ENABLED="true"
            shift
            ;;
        --scale)
            SCALE="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            break
            ;;
    esac
done

# Main command handling
case "$1" in
    dev)
        check_prerequisites
        deploy_dev
        ;;
    prod)
        check_prerequisites
        deploy_prod
        ;;
    gpu)
        check_prerequisites
        deploy_gpu
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    logs)
        show_logs "$@"
        ;;
    status)
        show_status
        ;;
    clean)
        clean_up
        ;;
    backup)
        backup_data
        ;;
    restore)
        restore_data "$@"
        ;;
    --help|help)
        show_help
        ;;
    *)
        log_error "Unknown command: $1"
        echo
        show_help
        exit 1
        ;;
esac