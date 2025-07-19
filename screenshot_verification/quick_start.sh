#!/bin/bash

# 手机截图AI识别真伪系统 - 快速启动脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_message() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}  手机截图AI识别真伪系统${NC}"
    echo -e "${BLUE}================================${NC}"
}

# 检查依赖
check_dependencies() {
    print_message "检查系统依赖..."
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3.8+ 未安装"
        exit 1
    fi
    
    # 检查pip
    if ! command -v pip3 &> /dev/null; then
        print_error "pip3 未安装"
        exit 1
    fi
    
    # 检查Docker（可选）
    if command -v docker &> /dev/null; then
        DOCKER_AVAILABLE=true
        print_message "Docker 已安装"
    else
        DOCKER_AVAILABLE=false
        print_warning "Docker 未安装，将使用本地模式"
    fi
    
    print_message "依赖检查完成"
}

# 安装Python依赖
install_python_deps() {
    print_message "安装Python依赖..."
    
    if [ -f "requirements.txt" ]; then
        pip3 install -r requirements.txt
        print_message "Python依赖安装完成"
    else
        print_error "requirements.txt 文件不存在"
        exit 1
    fi
}

# 创建必要的目录
create_directories() {
    print_message "创建必要的目录..."
    
    mkdir -p uploads temp models logs
    print_message "目录创建完成"
}

# 启动本地服务
start_local_service() {
    print_message "启动本地服务..."
    
    # 检查是否已有服务在运行
    if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
        print_warning "端口8000已被占用，请先停止其他服务"
        exit 1
    fi
    
    # 启动服务
    python3 scripts/start_server.py --debug &
    SERVER_PID=$!
    
    # 等待服务启动
    sleep 5
    
    # 检查服务是否启动成功
    if curl -f http://localhost:8000/ping >/dev/null 2>&1; then
        print_message "服务启动成功！"
        print_message "API地址: http://localhost:8000"
        print_message "文档地址: http://localhost:8000/docs"
        print_message "按 Ctrl+C 停止服务"
        
        # 保存PID
        echo $SERVER_PID > .server.pid
        
        # 等待用户中断
        wait $SERVER_PID
    else
        print_error "服务启动失败"
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi
}

# 启动Docker服务
start_docker_service() {
    print_message "启动Docker服务..."
    
    if [ "$DOCKER_AVAILABLE" = false ]; then
        print_error "Docker 不可用，无法启动Docker服务"
        exit 1
    fi
    
    # 构建并启动服务
    docker-compose up -d
    
    # 等待服务启动
    print_message "等待服务启动..."
    sleep 10
    
    # 检查服务状态
    if curl -f http://localhost:8000/ping >/dev/null 2>&1; then
        print_message "Docker服务启动成功！"
        print_message "API地址: http://localhost:8000"
        print_message "文档地址: http://localhost:8000/docs"
        print_message "使用 'docker-compose down' 停止服务"
    else
        print_error "Docker服务启动失败"
        docker-compose logs
        exit 1
    fi
}

# 运行测试
run_tests() {
    print_message "运行测试..."
    
    if command -v pytest &> /dev/null; then
        pytest tests/ -v
        print_message "测试完成"
    else
        print_warning "pytest 未安装，跳过测试"
    fi
}

# 运行示例客户端
run_example_client() {
    print_message "运行示例客户端..."
    
    if [ -f "examples/client_example.py" ]; then
        python3 examples/client_example.py --create-test
        print_message "示例客户端运行完成"
    else
        print_warning "示例客户端文件不存在"
    fi
}

# 停止服务
stop_service() {
    print_message "停止服务..."
    
    # 停止本地服务
    if [ -f ".server.pid" ]; then
        PID=$(cat .server.pid)
        if kill -0 $PID 2>/dev/null; then
            kill $PID
            print_message "本地服务已停止"
        fi
        rm -f .server.pid
    fi
    
    # 停止Docker服务
    if [ "$DOCKER_AVAILABLE" = true ]; then
        docker-compose down 2>/dev/null || true
        print_message "Docker服务已停止"
    fi
}

# 清理
cleanup() {
    print_message "清理资源..."
    
    # 停止服务
    stop_service
    
    # 清理临时文件
    rm -f .server.pid
    
    print_message "清理完成"
}

# 显示帮助
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  start       启动服务"
    echo "  stop        停止服务"
    echo "  restart     重启服务"
    echo "  test        运行测试"
    echo "  example     运行示例客户端"
    echo "  docker      使用Docker启动服务"
    echo "  local       使用本地模式启动服务"
    echo "  install     安装依赖"
    echo "  clean       清理资源"
    echo "  help        显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 start     # 启动服务"
    echo "  $0 docker    # 使用Docker启动服务"
    echo "  $0 test      # 运行测试"
}

# 主函数
main() {
    print_header
    
    # 检查参数
    case "${1:-start}" in
        "start")
            check_dependencies
            create_directories
            if [ "$DOCKER_AVAILABLE" = true ]; then
                start_docker_service
            else
                install_python_deps
                start_local_service
            fi
            ;;
        "stop")
            stop_service
            ;;
        "restart")
            stop_service
            sleep 2
            $0 start
            ;;
        "test")
            check_dependencies
            install_python_deps
            run_tests
            ;;
        "example")
            check_dependencies
            install_python_deps
            run_example_client
            ;;
        "docker")
            if [ "$DOCKER_AVAILABLE" = true ]; then
                start_docker_service
            else
                print_error "Docker 不可用"
                exit 1
            fi
            ;;
        "local")
            check_dependencies
            install_python_deps
            create_directories
            start_local_service
            ;;
        "install")
            check_dependencies
            install_python_deps
            create_directories
            ;;
        "clean")
            cleanup
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_error "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
}

# 捕获中断信号
trap cleanup INT TERM

# 运行主函数
main "$@"