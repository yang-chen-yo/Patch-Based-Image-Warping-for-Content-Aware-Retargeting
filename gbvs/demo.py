from saliency_models import gbvs, ittikochneibur, cas
import cv2
import sys
import time
from matplotlib import pyplot as plt

ALGORITHMS = {
    'gbvs': gbvs.compute_saliency,
    'ikn': ittikochneibur.compute_saliency,
    'cas': cas.compute_saliency,
}

if __name__ == '__main__':
    imname = sys.argv[1] if len(sys.argv) > 1 else "./images/bb.jpg"
    algo = sys.argv[2] if len(sys.argv) > 2 else "gbvs"  # which map gets saved/used downstream
    if algo not in ALGORITHMS:
        print("Unknown algorithm '{}', choose from: {}".format(algo, ', '.join(ALGORITHMS)))
        sys.exit(1)
    print("processing {} with {}".format(imname, algo))

    # 讀取圖片
    img = cv2.imread(imname)

    # 確保圖片正確讀取
    if img is None:
        print("Failed to load image!")
    else:
        # 計算三種顯著性圖，方便互相比較
        print("image size: {} x {}".format(img.shape[1], img.shape[0]))
        results = {}
        for name, compute in ALGORITHMS.items():
            start = time.time()
            results[name] = compute(img)
            print("saliency time ({}): {:.3f} s".format(name, time.time() - start))
        saliency_map_gbvs, saliency_map_ikn, saliency_map_cas = results['gbvs'], results['ikn'], results['cas']

        # 將選定演算法的結果保存
        oname = "./outputs/my_image_out{}.jpg".format(time.time())
        cv2.imwrite(oname, results[algo])
        print("SALIENCY_OUTPUT={}".format(oname))

        # 顯示圖片
        fig = plt.figure(figsize=(12, 3))

        # 原圖
        fig.add_subplot(1, 4, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # 注意轉換色彩空間
        plt.gca().set_title("Original Image")
        plt.axis('off')

        # GBVS 顯著性圖
        fig.add_subplot(1, 4, 2)
        plt.imshow(saliency_map_gbvs, cmap='gray')
        plt.gca().set_title("GBVS" + (" *" if algo == 'gbvs' else ""))
        plt.axis('off')

        # Itti-Koch-Neibur 顯著性圖
        fig.add_subplot(1, 4, 3)
        plt.imshow(saliency_map_ikn, cmap='gray')
        plt.gca().set_title("Itti Koch Neibur" + (" *" if algo == 'ikn' else ""))
        plt.axis('off')

        # Context-Aware Saliency 顯著性圖
        fig.add_subplot(1, 4, 4)
        plt.imshow(saliency_map_cas, cmap='gray')
        plt.gca().set_title("Context-Aware (CAS)" + (" *" if algo == 'cas' else ""))
        plt.axis('off')

        plt.show()
