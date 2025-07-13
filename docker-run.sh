#!/bin/bash

# Docker run script for vector-db-benchmark
# This script provides convenient ways to run the benchmark in Docker

set -e

# Default values
IMAGE_NAME="filipe958/vector-db-benchmark:latest"
REDIS_HOST="localhost"
REDIS_PORT="6379"
ENGINES="redis"
DATASET="random-100"
EXPERIMENT="redis-default-simple"
NETWORK=""
EXTRA_ARGS=""

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

print_example() {
    echo -e "${BLUE}[EXAMPLE]${NC} $1"
}

# Function to show usage
usage() {
    echo "Usage: $0 [OPTIONS] [-- EXTRA_ARGS]"
    echo ""
    echo "Options:"
    echo "  -i, --image IMAGE     Docker image name (default: redis-performance/vector-db-benchmark:latest)"
    echo "  -H, --host HOST       Redis host (default: localhost)"
    echo "  -p, --port PORT       Redis port (default: 6379)"
    echo "  -e, --engines ENGINES Engines to test (default: redis)"
    echo "  -d, --dataset DATASET Dataset to use (default: random-100)"
    echo "  -x, --experiment EXP  Experiment configuration (default: redis-default-simple)"
    echo "  -n, --network NET     Docker network to use"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Examples:"
    print_example "$0 # Run with defaults (help)"
    print_example "$0 -H redis -e redis -d random-100 -x redis-default-simple # Basic Redis benchmark"
    print_example "$0 -n redis-net -H redis-server # Use custom network"
    print_example "$0 -- --skip-upload --skip-search # Pass extra arguments"
    echo ""
    echo "Common Redis setups:"
    print_example "# Start Redis container first:"
    print_example "docker run -d --name redis-test -p 6379:6379 redis:8.2-rc1-bookworm"
    echo ""
    print_example "# Then run benchmark:"
    print_example "$0 -H localhost -e redis -d random-100"
    echo ""
    print_example "# With results output (mount current directory):"
    print_example "docker run --rm -v \$(pwd)/results:/app/results --network host filipe958/vector-db-benchmark:latest run.py --host localhost --engines redis"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--image)
            IMAGE_NAME="$2"
            shift 2
            ;;
        -H|--host)
            REDIS_HOST="$2"
            shift 2
            ;;
        -p|--port)
            REDIS_PORT="$2"
            shift 2
            ;;
        -e|--engines)
            ENGINES="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -x|--experiment)
            EXPERIMENT="$2"
            shift 2
            ;;
        -n|--network)
            NETWORK="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS="$*"
            break
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Build Docker run command
DOCKER_CMD="docker run --rm -it"

# Add network if specified
if [[ -n "$NETWORK" ]]; then
    DOCKER_CMD="$DOCKER_CMD --network $NETWORK"
    print_info "Using Docker network: $NETWORK"
fi

# Mount results directory
DOCKER_CMD="$DOCKER_CMD -v \$(pwd)/results:/app/results"

# Add image
DOCKER_CMD="$DOCKER_CMD $IMAGE_NAME"

# If no extra args provided, show help
if [[ -z "$EXTRA_ARGS" && "$ENGINES" == "redis" && "$DATASET" == "random-100" && "$EXPERIMENT" == "redis-default-simple" && "$REDIS_HOST" == "localhost" ]]; then
    print_info "No specific configuration provided, showing help:"
    DOCKER_CMD="$DOCKER_CMD run.py --help"
else
    # Add benchmark arguments
    DOCKER_CMD="$DOCKER_CMD run.py --host $REDIS_HOST --engines $ENGINES --dataset $DATASET --experiment $EXPERIMENT"

    # Add extra arguments if provided
    if [[ -n "$EXTRA_ARGS" ]]; then
        DOCKER_CMD="$DOCKER_CMD $EXTRA_ARGS"
    fi

    print_info "Configuration:"
    print_info "  Redis: $REDIS_HOST:$REDIS_PORT"
    print_info "  Engines: $ENGINES"
    print_info "  Dataset: $DATASET"
    print_info "  Experiment: $EXPERIMENT"
    if [[ -n "$EXTRA_ARGS" ]]; then
        print_info "  Extra args: $EXTRA_ARGS"
    fi
fi

print_info "Executing: $DOCKER_CMD"
echo ""

# Execute the command
eval $DOCKER_CMD
