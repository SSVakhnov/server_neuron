import cv2 as cv
import numpy as np
import pytesseract
import imutils
from skimage.filters import threshold_local

VALID_PERCENTAGE = 25

PATH_TO_TEMPLATE = 'forms/ua_blank_bio_id_edited.jpg'

# point means (width, height)
# first rect point # second rect point # data type # field name
global_roi = [
    [(350, 110), (800, 180), 'text', 'last_name'],
    [(350, 190), (800, 260), 'text', 'first_name'],
    [(350, 270), (800, 305), 'text', 'patronymic'],
    [(350, 315), (550, 360), 'text', 'sex'],
    [(350, 370), (550, 410), 'text', 'date_of_birth'],
    [(350, 420), (550, 460), 'text', 'date_of_expire'],
    [(570, 315), (800, 360), 'text', 'nationality'],
    [(570, 370), (800, 410), 'text', 'record_number'],
    [(570, 420), (800, 460), 'text', 'document_number'],
]


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def recognize_text(image_source):
    id_card_data = []
    for i, line in enumerate(global_roi):
        # вырезаем область на распознавание
        img_gray_crop = image_source[line[0][1]:line[1][1], line[0][0]:line[1][0]]
        # img_gray_crop = cv.resize(img_gray_crop, (520, 128), interpolation=cv.INTER_CUBIC)

        if line[2] == 'text':
            if line[3] in ['record_number', 'document_number', 'date_of_expire', 'date_of_birth']:
                text = pytesseract.image_to_string(
                    img_gray_crop,
                    lang="eng+rus",
                    config="--psm 4 --oem 3 -c tessedit_char_whitelist=0123456789")
            elif line[3] in ['sex']:
                text = pytesseract.image_to_string(
                    img_gray_crop,
                    lang="eng+rus",
                    config="--psm 4 --oem 3 -c tessedit_char_whitelist=FM")
            else:
                text = pytesseract.image_to_string(
                    img_gray_crop,
                    lang="eng+rus",
                    config="--psm 4 --oem 3 -c tessedit_char_whitelist=-ABCDEFGHIJKLMNOPQRSTUVWXYZ/")

            id_card_data.append(text)

    return id_card_data


def edged_image(img_gray):
    image_blur = cv.GaussianBlur(img_gray, (9, 9), 0)
    t = cv.Canny(image_blur, 0, 100)
    cv.imshow("Edge Detected Image", t)
    cv.waitKey(0)
    return t


def area_boundaries(img_source, img_edged):
    # найти контуры на обрезанном изображении, рационально организовать область
    # оставить только большие варианты
    all_contours = cv.findContours(img_edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    all_contours = imutils.grab_contours(all_contours)

    # сортировка контуров области по уменьшению и сохранение топ-1
    all_contours = sorted(all_contours, key=cv.contourArea, reverse=True)[:1]

    # aппроксимация контура
    perimeter = cv.arcLength(all_contours[0], True)
    ROI_dimensions = cv.approxPolyDP(all_contours[0], 0.02 * perimeter, True)

    # показать контуры на изображении
    cv.drawContours(img_source, [ROI_dimensions], -1, (0, 255, 0), 2)

    return ROI_dimensions


def get_document_edges(roi_dimensions):
    # изменение массива координат
    roi_dimensions = roi_dimensions.reshape(4, 2)

    # список удержания координат ROI
    rect = np.zeros((4, 2), dtype='float32')

    # наименьшая сумма будет у верхнего левого угла,
    # наибольшая — у нижнего правого угла
    s = np.sum(roi_dimensions, axis=1)
    rect[0] = roi_dimensions[np.argmin(s)]
    rect[2] = roi_dimensions[np.argmax(s)]

    # верх-право будет с минимальной разницей
    # низ-лево будет иметь максимальную разницу
    diff = np.diff(roi_dimensions, axis=1)
    rect[1] = roi_dimensions[np.argmin(diff)]
    rect[3] = roi_dimensions[np.argmax(diff)]

    # верх-лево, верх-право, низ-право, низ-лево
    (tl, tr, br, bl) = rect

    # вычислить ширину ROI
    width_a = np.sqrt((tl[0] - tr[0]) ** 2 + (tl[1] - tr[1]) ** 2)
    width_b = np.sqrt((bl[0] - br[0]) ** 2 + (bl[1] - br[1]) ** 2)
    width_max = max(int(width_a), int(width_b))

    # вычислить высоту ROI
    height_a = np.sqrt((tl[0] - bl[0]) ** 2 + (tl[1] - bl[1]) ** 2)
    height_b = np.sqrt((tr[0] - br[0]) ** 2 + (tr[1] - br[1]) ** 2)
    height_max = max(int(height_a), int(height_b))

    return rect, width_max, height_max


def prospective_transformation(source, rect, max_width, max_height):
    # набор итоговых точек для обзора всего документа
    # размер нового изображения
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    # вычислить матрицу перспективного преобразования и применить её
    transform_matrix = cv.getPerspectiveTransform(rect, dst)

    # преобразовать ROI
    return cv.warpPerspective(source, transform_matrix, (max_width, max_height))


def run():
    blank_img = cv.imread(PATH_TO_TEMPLATE)

    blank_gray = cv.cvtColor(blank_img, cv.COLOR_BGR2GRAY)
    blank_gray = cv.threshold(blank_gray, 100, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

    orb = cv.ORB_create(1000)
    key_points_1, descriptions_1 = orb.detectAndCompute(blank_gray, None)
    blank_gray_with_key_points = cv.drawKeypoints(blank_gray, key_points_1, None)

    # то же самое, но уже не для бланка, а для изображения на обработку
    img = cv.imread('forms/ua_bio_id_front.jpg')

    img = unsharp_mask(img)

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edged_img = edged_image(img_gray)
    ROI_dimensions = area_boundaries(img, edged_img)
    # cv.imshow("Edge Detected Image", img)
    # cv.waitKey(0)
    img_gray = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 85, 25)

    # img = cv.resize(img, (640, 480), interpolation=cv.INTER_AREA)
    # img_gray = cv.resize(img_gray, (640, 480), interpolation=cv.INTER_AREA)
    # edged_img = edged_image(img_gray)
    # area_boundaries(img, edged_img)
    cv.imshow("Edge Detected Image", img)
    cv.waitKey(0)

    rect, max_width, max_height = get_document_edges(ROI_dimensions)

    scan = prospective_transformation(img_gray, rect, max_width, max_height)
    cv.imshow("Scaned", scan)
    cv.waitKey(0)

    # img_gray = img
    blank_gray_height, blank_gray_width = blank_gray.shape[:2]
    to_size = (blank_gray_width, blank_gray_height)
    scan = cv.resize(scan, to_size, interpolation=cv.INTER_AREA)

    id_card_data = recognize_text(scan)

    print(id_card_data)

    cv.imshow('Filled rectangles', scan)
    cv.waitKey(0)


if __name__ == '__main__':
    run()
