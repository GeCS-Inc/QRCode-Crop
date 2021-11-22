import cv2, os
import numpy as np
from PIL import Image, ImageDraw

OUTPUT_DIR = "outputs"

def detect_qr(img_path, warp_type = "four-corners"):
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    img_orig = Image.open(img_path).convert("RGB")
    img = np.array(img_orig)

    img_out = img_orig.copy()
    draw    = ImageDraw.Draw(img_out)
    
    qrd = cv2.QRCodeDetector()

    retval, decoded_info, points, straight_qrcode = qrd.detectAndDecodeMulti(img)
    if not retval: return

    zipped = sorted(zip(decoded_info, points),key=lambda x: x[0])
    for info, point in zipped:
        for x, y in point:
            r = 10
            draw.ellipse((x-r, y-r, x+r, y+r), fill=(255, 0, 0), outline=(0, 0, 0))
    
    w, h = 500, 500
    src, dst = None, None

    if warp_type == "four-corners" and len(zipped) == 4:
        assert '0' in decoded_info
        assert '1' in decoded_info
        assert '2' in decoded_info
        assert '3' in decoded_info

        p0, p1, p2, p3 = [p for _, p in zipped]
        p0, p1, p2, p3 = p0[0], p1[1], p2[3], p3[2]
        src = np.float32([p0, p1, p2, p3])
        dst = np.float32([[0,0], [w,0], [0,h], [w,h]])
    if warp_type == "two-corners" and len(zipped) == 2:
        assert '0' in decoded_info
        assert '3' in decoded_info

        p0, p3 = [p for _, p in zipped]
        p1 = cross_point((p0[0], p0[1]), (p3[2], p3[1]))
        p2 = cross_point((p0[0], p0[3]), (p3[2], p3[3]))
        p0, p3 = p0[0], p3[2]
        src = np.float32([p0, p1, p2, p3])
        dst = np.float32([[0,0], [w,0], [0,h], [w,h]])

        for x, y in [p0, p1, p2, p3]:
            r = 10
            draw.ellipse((x-r, y-r, x+r, y+r), fill=(255, 0, 0), outline=(0, 0, 0))

    if src is None or dst is None:
        print(f"検知したQRコードの数: {len(zipped)}")
        return

    # 変換行列
    M = cv2.getPerspectiveTransform(src, dst)
    # 射影変換・透視変換する
    output = cv2.warpPerspective(np.array(img_orig), M,(w, h))

    output_dir = os.path.join(OUTPUT_DIR, img_name)
    os.makedirs(output_dir, exist_ok=True)
    img_out.save(os.path.join(output_dir, "qr_detected.png"))
    Image.fromarray(output).save(os.path.join(output_dir, "qr_cropped.png"))

def cross_point(l1, l2):
    (p1x, p1y), (p3x, p3y) = l1
    (p2x, p2y), (p4x, p4y) = l2

    s1 = (p4x - p2x) * (p1y - p2y) - (p4y - p2y) * (p1x - p2x)
    s2 = (p4x - p2x) * (p2y - p3y) - (p4y - p2y) * (p2x - p3x)

    cp = np.array([
        p1x + (p3x - p1x) * s1 / (s1 + s2),
        p1y + (p3y - p1y) * s1 / (s1 + s2)
    ])

    return cp

if __name__ == '__main__':
    detect_qr('./samples/qr_test1.png', warp_type='four-corners')
    detect_qr('./samples/qr_test2.png', warp_type='four-corners')
    detect_qr('./samples/qr_test3.png', warp_type='two-corners')
    detect_qr('./samples/qr_test4.png', warp_type='two-corners')
    # detect_qr('./samples/qr_test5.png', warp_type='two-corners') # Error
    detect_qr('./samples/qr_test6.png', warp_type='two-corners')