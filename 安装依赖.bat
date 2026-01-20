@echo off
chcp 65001 >nul
title pyMediaTools 依赖安装程序
color 0A

echo.
echo ==========================================
echo    pyMediaTools 一键安装依赖
echo ==========================================
echo.
echo 正在检查 Python...

python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [错误] 未找到 Python！
    echo.
    echo 请先下载安装 Python：
    echo https://www.python.org/downloads/
    echo.
    echo 安装时请勾选 "Add Python to PATH"
    echo.
    pause
    exit /b 1
)

echo [OK] Python 已安装
echo.

echo 正在检查 FFmpeg...
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [警告] 未找到 FFmpeg
    echo 部分功能可能无法使用（如视频转换）
    echo.
) else (
    echo [OK] FFmpeg 已安装
)
echo.

echo 正在安装 Python 依赖包...
echo 请稍候，这可能需要几分钟...
echo.

pip install flask flask-cors requests pydub pysrt yt-dlp diff-match-patch -i https://pypi.tuna.tsinghua.edu.cn/simple --quiet

if errorlevel 1 (
    echo.
    echo [错误] 安装失败，正在尝试备用源...
    pip install flask flask-cors requests pydub pysrt yt-dlp diff-match-patch --quiet
)

echo.
echo ==========================================
echo    安装完成！
echo ==========================================
echo.
echo 现在可以运行 pyMediaTools.exe 了
echo.
pause
