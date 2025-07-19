#!/bin/bash

# 📱 移动截图AI真实性检测系统 - 完整快速开始脚本
# 自动化执行从环境准备到生产部署的完整流程

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
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

# 检查命令是否存在
check_command() {
    if ! command -v $1 &> /dev/null; then
        log_error "$1 未安装，请先安装 $1"
        exit 1
    fi
}

# 显示帮助信息
show_help() {
    echo "📱 移动截图AI真实性检测系统 - 快速开始脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --help, -h              显示此帮助信息"
    echo "  --env ENV               指定环境 (development/testing/production)"
    echo "  --data-only             仅执行数据收集和准备"
    echo "  --train-only            仅执行模型训练"
    echo "  --deploy-only           仅执行部署"
    echo "  --test-only             仅执行测试"
    echo "  --monitor-only          仅执行监控设置"
    echo "  --skip-data             跳过数据收集"
    echo "  --skip-train            跳过模型训练"
    echo "  --skip-test             跳过测试"
    echo "  --skip-deploy           跳过部署"
    echo "  --clean                 清理所有生成的文件"
    echo ""
    echo "示例:"
    echo "  $0 --env development    完整开发环境设置"
    echo "  $0 --data-only          仅收集和准备数据"
    echo "  $0 --train-only         仅训练模型"
    echo "  $0 --deploy-only        仅部署服务"
    echo "  $0 --clean              清理所有文件"
}

# 解析命令行参数
ENV="development"
DATA_ONLY=false
TRAIN_ONLY=false
DEPLOY_ONLY=false
TEST_ONLY=false
MONITOR_ONLY=false
SKIP_DATA=false
SKIP_TRAIN=false
SKIP_TEST=false
SKIP_DEPLOY=false
CLEAN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help
            exit 0
            ;;
        --env)
            ENV="$2"
            shift 2
            ;;
        --data-only)
            DATA_ONLY=true
            shift
            ;;
        --train-only)
            TRAIN_ONLY=true
            shift
            ;;
        --deploy-only)
            DEPLOY_ONLY=true
            shift
            ;;
        --test-only)
            TEST_ONLY=true
            shift
            ;;
        --monitor-only)
            MONITOR_ONLY=true
            shift
            ;;
        --skip-data)
            SKIP_DATA=true
            shift
            ;;
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        --skip-test)
            SKIP_TEST=true
            shift
            ;;
        --skip-deploy)
            SKIP_DEPLOY=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        *)
            log_error "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

log_info "项目根目录: $PROJECT_ROOT"
log_info "目标环境: $ENV"

# 清理函数
cleanup() {
    log_info "清理生成的文件..."
    
    # 清理数据目录
    if [ -d "data" ]; then
        rm -rf data/screenshots/*
        rm -rf data/cache/*
        log_success "数据目录已清理"
    fi
    
    # 清理输出目录
    if [ -d "outputs" ]; then
        rm -rf outputs/*
        log_success "输出目录已清理"
    fi
    
    # 清理模型目录
    if [ -d "models" ]; then
        rm -rf models/cache/*
        log_success "模型缓存已清理"
    fi
    
    # 清理日志
    if [ -d "logs" ]; then
        rm -rf logs/*
        log_success "日志已清理"
    fi
    
    # 清理Docker容器和镜像
    if command -v docker &> /dev/null; then
        docker stop $(docker ps -q --filter "name=screenshot-verification") 2>/dev/null || true
        docker rm $(docker ps -aq --filter "name=screenshot-verification") 2>/dev/null || true
        docker rmi screenshot-verification:latest 2>/dev/null || true
        log_success "Docker资源已清理"
    fi
    
    log_success "清理完成"
}

# 如果只是清理，执行后退出
if [ "$CLEAN" = true ]; then
    cleanup
    exit 0
fi

# 检查系统要求
check_requirements() {
    log_info "检查系统要求..."
    
    # 检查Python版本
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 未安装"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    log_info "Python 版本: $PYTHON_VERSION"
    
    # 检查Docker
    if command -v docker &> /dev/null; then
        log_success "Docker 已安装"
    else
        log_warning "Docker 未安装，将使用本地模式"
    fi
    
    # 检查系统资源
    MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$MEMORY_GB" -lt 8 ]; then
        log_warning "内存不足，推荐至少 8GB，当前: ${MEMORY_GB}GB"
    else
        log_success "内存: ${MEMORY_GB}GB"
    fi
    
    # 检查磁盘空间
    DISK_GB=$(df -BG . | awk 'NR==2{print $4}' | sed 's/G//')
    if [ "$DISK_GB" -lt 50 ]; then
        log_warning "磁盘空间不足，推荐至少 50GB，当前: ${DISK_GB}GB"
    else
        log_success "可用磁盘空间: ${DISK_GB}GB"
    fi
    
    log_success "系统要求检查完成"
}

# 环境准备
setup_environment() {
    log_info "准备环境..."
    
    # 创建必要的目录
    mkdir -p data/screenshots/{authentic,fake}
    mkdir -p data/cache
    mkdir -p models/cache
    mkdir -p outputs
    mkdir -p logs
    mkdir -p config
    
    # 创建虚拟环境
    if [ ! -d "venv" ]; then
        log_info "创建Python虚拟环境..."
        python3 -m venv venv
    fi
    
    # 激活虚拟环境
    source venv/bin/activate
    
    # 升级pip
    pip install --upgrade pip
    
    # 安装依赖
    log_info "安装Python依赖..."
    pip install -r requirements.txt
    
    # 安装额外依赖
    log_info "安装训练相关依赖..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install optuna albumentations schedule psutil
    
    log_success "环境准备完成"
}

# 数据收集和准备
collect_data() {
    if [ "$SKIP_DATA" = true ]; then
        log_info "跳过数据收集"
        return
    fi
    
    log_info "开始数据收集和准备..."
    
    # 激活虚拟环境
    source venv/bin/activate
    
    # 收集真实截图
    log_info "收集真实截图..."
    python scripts/data_collection.py \
        --output_dir ./data/screenshots \
        --num_authentic 500 \
        --collect_only
    
    # 生成伪造截图
    log_info "生成伪造截图..."
    python scripts/data_collection.py \
        --output_dir ./data/screenshots \
        --num_fake 500 \
        --generate_only
    
    # 构建数据集
    log_info "构建数据集..."
    python scripts/data_collection.py \
        --output_dir ./data/screenshots
    
    # 验证数据集
    log_info "验证数据集..."
    python scripts/data_collection.py \
        --output_dir ./data/screenshots \
        --validate_only
    
    log_success "数据收集和准备完成"
}

# 模型训练
train_models() {
    if [ "$SKIP_TRAIN" = true ]; then
        log_info "跳过模型训练"
        return
    fi
    
    log_info "开始模型训练..."
    
    # 激活虚拟环境
    source venv/bin/activate
    
    # 超参数优化
    log_info "执行超参数优化..."
    python scripts/train_and_optimize.py \
        --mode optimize \
        --data_dir ./data/screenshots \
        --output_dir ./outputs \
        --model_type efficientnet \
        --n_trials 20
    
    # 模型选择
    log_info "执行模型选择..."
    python scripts/train_and_optimize.py \
        --mode select \
        --data_dir ./data/screenshots \
        --output_dir ./outputs
    
    # 完整训练
    log_info "执行完整训练..."
    python scripts/train_and_optimize.py \
        --mode train \
        --data_dir ./data/screenshots \
        --output_dir ./outputs
    
    # 性能优化
    log_info "执行性能优化..."
    if [ -f "outputs/best_model.pth" ]; then
        python scripts/train_and_optimize.py \
            --mode optimize_performance \
            --model_path ./outputs/best_model.pth \
            --output_dir ./outputs/optimized
    fi
    
    log_success "模型训练完成"
}

# 启动服务
start_service() {
    log_info "启动服务..."
    
    # 激活虚拟环境
    source venv/bin/activate
    
    # 检查是否有训练好的模型
    if [ ! -f "outputs/best_model.pth" ] && [ ! -f "models/best_model.pth" ]; then
        log_warning "未找到训练好的模型，将使用默认模型"
    fi
    
    # 启动服务
    log_info "启动API服务..."
    python main.py --env $ENV &
    SERVICE_PID=$!
    
    # 等待服务启动
    sleep 10
    
    # 检查服务状态
    if curl -s http://localhost:8000/api/v1/health > /dev/null; then
        log_success "服务启动成功"
        echo $SERVICE_PID > .service.pid
    else
        log_error "服务启动失败"
        exit 1
    fi
}

# 运行测试
run_tests() {
    if [ "$SKIP_TEST" = true ]; then
        log_info "跳过测试"
        return
    fi
    
    log_info "开始运行测试..."
    
    # 激活虚拟环境
    source venv/bin/activate
    
    # 等待服务完全启动
    sleep 5
    
    # 运行性能测试
    log_info "运行性能测试..."
    python scripts/deploy_and_test.py \
        --mode test \
        --config config/test_config.json \
        --base_url http://localhost:8000
    
    # 运行API测试
    log_info "运行API测试..."
    python -m pytest tests/test_api.py -v
    
    log_success "测试完成"
}

# 部署服务
deploy_service() {
    if [ "$SKIP_DEPLOY" = true ]; then
        log_info "跳过部署"
        return
    fi
    
    log_info "开始部署..."
    
    # 停止现有服务
    if [ -f ".service.pid" ]; then
        kill $(cat .service.pid) 2>/dev/null || true
        rm .service.pid
    fi
    
    # 根据环境选择部署方式
    case $ENV in
        "development")
            log_info "开发环境部署..."
            if command -v docker-compose &> /dev/null; then
                docker-compose -f docker-compose.dev.yml up -d
            else
                start_service
            fi
            ;;
        "testing")
            log_info "测试环境部署..."
            if command -v docker-compose &> /dev/null; then
                docker-compose -f docker-compose.test.yml up -d
            else
                start_service
            fi
            ;;
        "production")
            log_info "生产环境部署..."
            if command -v docker-compose &> /dev/null; then
                docker-compose -f docker-compose.prod.yml up -d
            else
                log_error "生产环境需要Docker支持"
                exit 1
            fi
            ;;
        *)
            log_error "未知环境: $ENV"
            exit 1
            ;;
    esac
    
    log_success "部署完成"
}

# 设置监控
setup_monitoring() {
    log_info "设置监控..."
    
    # 激活虚拟环境
    source venv/bin/activate
    
    # 启动监控
    python scripts/deploy_and_test.py \
        --mode monitor \
        --config config/test_config.json &
    MONITOR_PID=$!
    
    echo $MONITOR_PID > .monitor.pid
    
    log_success "监控设置完成"
}

# 启动持续改进
start_improvement() {
    log_info "启动持续改进系统..."
    
    # 激活虚拟环境
    source venv/bin/activate
    
    # 启动反馈收集
    python scripts/continuous_improvement.py \
        --mode collect \
        --config config/improvement_config.json &
    FEEDBACK_PID=$!
    
    # 启动性能监控
    python scripts/continuous_improvement.py \
        --mode monitor \
        --config config/improvement_config.json &
    PERF_PID=$!
    
    echo $FEEDBACK_PID > .feedback.pid
    echo $PERF_PID > .perf.pid
    
    log_success "持续改进系统启动完成"
}

# 显示状态
show_status() {
    log_info "系统状态:"
    
    # 检查服务状态
    if curl -s http://localhost:8000/api/v1/health > /dev/null; then
        log_success "API服务: 运行中"
        echo "API地址: http://localhost:8000"
        echo "健康检查: http://localhost:8000/api/v1/health"
        echo "API文档: http://localhost:8000/docs"
    else
        log_error "API服务: 未运行"
    fi
    
    # 检查监控状态
    if [ -f ".monitor.pid" ] && kill -0 $(cat .monitor.pid) 2>/dev/null; then
        log_success "监控服务: 运行中"
        echo "Prometheus: http://localhost:8001"
    else
        log_warning "监控服务: 未运行"
    fi
    
    # 显示数据统计
    if [ -d "data/screenshots" ]; then
        AUTHENTIC_COUNT=$(find data/screenshots/authentic -name "*.png" 2>/dev/null | wc -l)
        FAKE_COUNT=$(find data/screenshots/fake -name "*.png" 2>/dev/null | wc -l)
        log_info "数据集统计:"
        echo "  真实截图: $AUTHENTIC_COUNT"
        echo "  伪造截图: $FAKE_COUNT"
    fi
    
    # 显示模型信息
    if [ -f "outputs/best_model.pth" ]; then
        log_success "模型: 已训练"
        echo "模型路径: outputs/best_model.pth"
    elif [ -f "models/best_model.pth" ]; then
        log_success "模型: 已加载"
        echo "模型路径: models/best_model.pth"
    else
        log_warning "模型: 未找到"
    fi
}

# 清理函数
cleanup_on_exit() {
    log_info "清理资源..."
    
    # 停止服务
    if [ -f ".service.pid" ]; then
        kill $(cat .service.pid) 2>/dev/null || true
        rm .service.pid
    fi
    
    # 停止监控
    if [ -f ".monitor.pid" ]; then
        kill $(cat .monitor.pid) 2>/dev/null || true
        rm .monitor.pid
    fi
    
    # 停止反馈收集
    if [ -f ".feedback.pid" ]; then
        kill $(cat .feedback.pid) 2>/dev/null || true
        rm .feedback.pid
    fi
    
    # 停止性能监控
    if [ -f ".perf.pid" ]; then
        kill $(cat .perf.pid) 2>/dev/null || true
        rm .perf.pid
    fi
}

# 设置退出时的清理
trap cleanup_on_exit EXIT

# 主函数
main() {
    log_info "🚀 开始移动截图AI真实性检测系统部署"
    
    # 检查系统要求
    check_requirements
    
    # 根据参数执行相应操作
    if [ "$DATA_ONLY" = true ]; then
        setup_environment
        collect_data
    elif [ "$TRAIN_ONLY" = true ]; then
        setup_environment
        train_models
    elif [ "$DEPLOY_ONLY" = true ]; then
        deploy_service
    elif [ "$TEST_ONLY" = true ]; then
        run_tests
    elif [ "$MONITOR_ONLY" = true ]; then
        setup_monitoring
    else
        # 完整流程
        setup_environment
        collect_data
        train_models
        deploy_service
        run_tests
        setup_monitoring
        start_improvement
    fi
    
    # 显示状态
    show_status
    
    log_success "🎉 部署完成！"
    log_info "下一步:"
    echo "  1. 访问 http://localhost:8000/docs 查看API文档"
    echo "  2. 运行示例: python examples/client_example.py"
    echo "  3. 查看日志: tail -f logs/app.log"
    echo "  4. 停止服务: $0 --clean"
}

# 执行主函数
main "$@"