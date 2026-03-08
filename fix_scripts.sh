#!/bin/bash
# 修复行尾符问题

echo "正在修复脚本文件的行尾符..."

for file in autodl_setup.sh autodl_start.sh autodl_monitor.sh; do
    if [ -f "$file" ]; then
        echo "修复 $file..."
        sed -i 's/\r$//' "$file"
        chmod +x "$file"
    fi
done

echo "✓ 修复完成！"
echo ""
echo "现在可以运行: bash autodl_setup.sh"
