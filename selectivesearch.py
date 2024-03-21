import sys
import cv2

def get_selective_search():
    gs = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    return gs


def config(gs, img, strategy='q'):
    gs.setBaseImage(img)

    if (strategy == 's'):
        gs.switchToSingleStrategy()
    elif (strategy == 'f'):
        gs.switchToSelectiveSearchFast()
    elif (strategy == 'q'):
        gs.switchToSelectiveSearchQuality()
    else:
        print(__doc__)
        sys.exit(1)

def get_rects(gs):
    rects = gs.process()
    rects[:, 2] += rects[:, 0]
    rects[:, 3] += rects[:, 1]

    return rects

def visualize_rects(image, rects):
    for x, y, w, h in rects:
        cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)  # 초록색으로 경계 상자를 그림

    # 시각화된 이미지를 표시
    cv2.imshow("Rectangles", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':

    gs = get_selective_search() # 선택적 검색 세그멘테이션을 초기화하여 이미지에서 객체를 검출하기 위한 준비를 수행

    img = cv2.imread('./my_voc2007/voc_car/train/JPEGImages/000012.jpg', cv2.IMREAD_COLOR)
    config(gs, img, strategy='q')

    rects = get_rects(gs)
    print(rects)

    visualize_rects(img, rects)