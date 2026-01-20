#!/bin/bash
# 将SAM3分割的mask图片合成为视频

# 参数
INPUT_DIR="./outputs/sam3_visualizations"
OUTPUT_VIDEO="${INPUT_DIR}/demo_masked_video.mp4"
FRAMERATE=30

# 检查目录
if [ ! -d "$INPUT_DIR" ]; then
    echo "错误: 目录 $INPUT_DIR 不存在"
    exit 1
fi

# 统计帧数
FRAME_COUNT=$(ls "$INPUT_DIR"/*_obj0.jpg 2>/dev/null | wc -l)
if [ $FRAME_COUNT -eq 0 ]; then
    echo "错误: 没有找到mask图片"
    exit 1
fi

echo "找到 $FRAME_COUNT 帧mask图片"

# 创建符号链接以便ffmpeg按顺序读取（因为文件名有空格）
TMP_DIR=$(mktemp -d)
cd "$INPUT_DIR"
i=0
for f in $(ls *_frame*_obj0.jpg | sort -V); do
    ln -s "$(pwd)/$f" "$TMP_DIR/frame_$(printf "%04d" $i).jpg"
    i=$((i+1))
done

echo "生成视频中..."
ffmpeg -framerate $FRAMERATE \
    -i "$TMP_DIR/frame_%04d.jpg" \
    -c:v libx264 \
    -pix_fmt yuv420p \
    -y "$OUTPUT_VIDEO" \
    2>&1 | grep -E "(frame=|Output)"

# 清理临时文件
rm -rf "$TMP_DIR"

if [ -f "$OUTPUT_VIDEO" ]; then
    echo "✓ 视频已生成: $OUTPUT_VIDEO"
    echo "视频信息:"
    ffprobe -v error -show_entries format=duration:stream=width,height,nb_frames -of default=noprint_wrappers=1 "$OUTPUT_VIDEO"
else
    echo "✗ 视频生成失败"
    exit 1
fi
