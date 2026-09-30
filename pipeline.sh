#!/usr/bin/env bash
# 一鍵跑完：Python 顯著圖 -> 複製進 res/ -> 編譯 -> 執行 C++ 變形
#
# 用法：
#   ./pipeline.sh                             # gallery.jpg + gbvs，輸出正方形（預設，跟之前行為一致）
#   ./pipeline.sh cat.jpg                     # cat.jpg + gbvs
#   ./pipeline.sh cat.jpg cas                 # cat.jpg + Context-Aware Saliency（論文原本指定的方法）
#   ./pipeline.sh cat.jpg ikn                 # cat.jpg + Itti-Koch-Niebur
#   ./pipeline.sh wide.jpg cas 1200 900       # 指定目標寬高（例如 16:9 來源縮成 4:3，而不是被硬擠成 1:1）
#
# 最長邊超過 MAX_SIDE（預設 1024，跟論文與 RetargetMe 一致）的圖會先等比例縮小，
# 原圖檔不會被修改。要改上限：MAX_SIDE=2048 ./pipeline.sh big.jpg cas
set -e  # 任何一步失敗就整個腳本停止，不要拿壞資料繼續往下跑

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_IMAGE="${1:-gallery.jpg}"
ALGO="${2:-gbvs}"
TARGET_WIDTH="${3:-}"
TARGET_HEIGHT="${4:-}"
MAX_SIDE="${MAX_SIDE:-1024}"

echo "== 1/4 準備輸入圖（最長邊上限 ${MAX_SIDE}） =="
cd "$PROJECT_ROOT"
mkdir -p res
# 大圖會讓 CPLEX 和 rendering 慢到幾十分鐘，而且 20 px 網格相對整張圖會變得過細。
# 沒超過上限就原檔複製；超過就等比例縮小，用 PNG（無損）寫進 res/gallery.jpg
# （OpenCV 讀檔看內容不看副檔名）。
"$PROJECT_ROOT/env/bin/python" - "gbvs/images/${SRC_IMAGE}" res/gallery.jpg "$MAX_SIDE" <<'PY'
import shutil, sys
import cv2
src, dst, max_side = sys.argv[1], sys.argv[2], int(sys.argv[3])
image = cv2.imread(src)
if image is None:
    sys.exit("讀不到圖片: " + src)
h, w = image.shape[:2]
if max(h, w) <= max_side:
    shutil.copy(src, dst)
    print("原圖 {} x {}，不需縮小".format(w, h))
else:
    scale = max_side / float(max(h, w))
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    small = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    cv2.imencode(".png", small)[1].tofile(dst)
    print("原圖 {} x {} -> 等比例縮小為 {} x {}".format(w, h, size[0], size[1]))
PY

echo "== 2/4 計算顯著圖（演算法: ${ALGO}） =="
cd "$PROJECT_ROOT/gbvs"
# 對 res/ 裡（可能已縮小）的圖算 saliency，確保兩者尺寸一致。
# MPLBACKEND=Agg：demo.py 裡的 plt.show() 平常會跳視窗、卡住等你關閉，
# 這裡不需要看那張圖，用非互動式後端讓它直接跳過不卡住。
SALIENCY_LINE=$(MPLBACKEND=Agg "$PROJECT_ROOT/env/bin/python" demo.py "$PROJECT_ROOT/res/gallery.jpg" "${ALGO}" | tee /dev/stderr | grep '^SALIENCY_OUTPUT=')
SALIENCY_FILE="${SALIENCY_LINE#SALIENCY_OUTPUT=}"
cd "$PROJECT_ROOT"
cp "gbvs/${SALIENCY_FILE#./}" res/gs.jpeg

echo "== 3/4 編譯 C++ =="
make

echo "== 4/4 執行變形 =="
if [ -n "$TARGET_WIDTH" ] && [ -n "$TARGET_HEIGHT" ]; then
    make run ARGS="$TARGET_WIDTH $TARGET_HEIGHT"
else
    make run
fi

echo "完成，結果在 result/result_gs.png"
