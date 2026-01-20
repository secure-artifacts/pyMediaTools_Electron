@echo off
chcp 65001 >nul
title pyMediaTools 后端服务
color 0A

echo.
echo ==========================================
echo    pyMediaTools 后端服务
echo ==========================================
echo.
echo 启动中，请稍候...
echo.

cd /d "%~dp0resources\backend"

python server.py

if errorlevel 1 (
    echo.
    echo [错误] 启动失败！
    echo.
    echo 可能原因：
    echo 1. 未安装 Python
    echo 2. 未安装必要的 Python 包
    echo.
    echo 请运行 安装依赖.bat 后重试
    echo.
    pause
)
