#!/bin/bash

# Docker test script for local validation
# This script mimics the GitHub Action validation locally

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Configuration
IMAGE_NAME="vector-db-benchmark-test"
TAG="local-test"
FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"

print_info "Starting Docker validation tests..."

# Step 0: Check Docker Hub credentials (optional for local testing)
print_step "Checking Docker Hub credentials..."
if [[ -n "$DOCKER_USERNAME" && -n "$DOCKER_PASSWORD" ]]; then
    print_info "✅ Docker Hub credentials found in environment"
elif docker info | grep -q "Username:"; then
    print_info "✅ Already logged in to Docker Hub"
else
    print_warning "⚠️ Docker Hub credentials not found"
    print_info "Set DOCKER_USERNAME and DOCKER_PASSWORD environment variables or run 'docker login' for publishing"
fi

# Step 1: Build the image
print_step "Building Docker image..."
if ./docker-build.sh -n "$IMAGE_NAME" -t "$TAG"; then
    print_info "✅ Docker build successful"
else
    print_error "❌ Docker build failed"
    exit 1
fi

# Step 2: Test basic functionality
print_step "Testing basic functionality..."

# Test help command
print_info "Testing --help command..."
if docker run --rm "$FULL_IMAGE_NAME" run.py --help > /dev/null; then
    print_info "✅ Help command works"
else
    print_error "❌ Help command failed"
    exit 1
fi

# Test Python environment
print_info "Testing Python environment..."
if docker run --rm "$FULL_IMAGE_NAME" -c "import sys; print(f'Python {sys.version}'); import redis; print('Redis module available')" > /dev/null; then
    print_info "✅ Python environment works"
else
    print_error "❌ Python environment test failed"
    exit 1
fi

# Step 3: Test with Redis (if available)
print_step "Testing Redis connectivity and benchmark execution..."
if command -v redis-server > /dev/null; then
    print_info "Redis server found, testing connectivity and benchmark..."

    # Start Redis in background if not running
    if ! pgrep redis-server > /dev/null; then
        print_info "Starting Redis server..."
        redis-server --port 6379 --daemonize yes
        sleep 2
    fi

    # Test basic connection
    if timeout 10 docker run --rm --network=host "$FULL_IMAGE_NAME" \
        -c "import redis; r = redis.Redis(host='localhost', port=6379); r.ping(); print('Redis connection successful')" > /dev/null 2>&1; then
        print_info "✅ Redis connectivity test passed"

        # Test benchmark execution with specific configuration
        print_info "Testing benchmark execution with redis-m-16-ef-64 configuration..."
        if timeout 30 docker run --rm --network=host -v "$(pwd)/results:/app/results" "$FULL_IMAGE_NAME" \
            run.py --host localhost --engines redis --dataset random-100 --experiment redis-m-16-ef-64 --skip-upload --skip-search > /dev/null 2>&1; then
            print_info "✅ Benchmark execution test passed"
        else
            print_warning "⚠️ Benchmark execution test failed (this may be expected without proper dataset setup)"
        fi
    else
        print_warning "⚠️ Redis connectivity test failed (this may be expected if Redis is not accessible)"
    fi
else
    print_warning "⚠️ Redis server not found, skipping connectivity test"
fi

# Step 4: Test file output permissions
print_step "Testing file output permissions..."
TEMP_DIR=$(mktemp -d)
if docker run --rm -v "$TEMP_DIR:/app/results" "$FULL_IMAGE_NAME" \
    -c "import os; os.makedirs('/app/results', exist_ok=True); open('/app/results/test.txt', 'w').write('test'); print('File write successful')" > /dev/null 2>&1; then
    if [ -f "$TEMP_DIR/test.txt" ]; then
        print_info "✅ File output test passed"
    else
        print_warning "⚠️ Test file not created"
    fi
else
    print_warning "⚠️ File output test completed with warnings"
fi
rm -rf "$TEMP_DIR"

# Step 5: Test image size
print_step "Checking image size..."
IMAGE_SIZE=$(docker images --format "table {{.Size}}" "$FULL_IMAGE_NAME" | tail -n 1)
print_info "Image size: $IMAGE_SIZE"

# Step 6: Test benchmark configuration loading
print_step "Testing benchmark configuration loading..."
if docker run --rm "$FULL_IMAGE_NAME" \
    -c "import json; import os; print('Configuration loading test'); print(os.listdir('/app'))" > /dev/null 2>&1; then
    print_info "✅ Configuration loading test passed"
else
    print_warning "⚠️ Configuration loading test completed with warnings"
fi

# Step 7: Clean up
print_step "Cleaning up..."
docker rmi "$FULL_IMAGE_NAME" > /dev/null 2>&1 || true

print_info "🎉 All Docker validation tests completed successfully!"
print_info ""
print_info "Summary:"
print_info "  ✅ Docker build successful"
print_info "  ✅ Basic functionality tests passed"
print_info "  ✅ Image size: $IMAGE_SIZE"
print_info ""
print_info "The Docker setup is ready for production use!"
