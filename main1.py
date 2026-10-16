
import time
import random
import re
import cv2
import numpy as np
import mss
import pytesseract
import keyboard

# Путь к установленному Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ================= НАСТРОЙКИ СКОРОСТИ И ОБЛАСТИ =================
TARGET_SPEED_CPM = 500  # Желаемая скорость печати (знаков в минуту)

# Координаты серого поля с текстом
MONITOR_ROI = {
    "top": 414,
    "left": 310,
    "width": 1022,
    "height": 271
}
# ================================================================

# Аппаратные скан-коды физических клавиш стандартной клавиатуры (не зависят от раскладки!)
RUS_TO_SCANCODE = {
    'й': 16, 'ц': 17, 'у': 18, 'к': 19, 'е': 20, 'н': 21, 'г': 22, 'ш': 23, 'щ': 24, 'з': 25, 'х': 26, 'ъ': 27,
    'ф': 30, 'ы': 31, 'в': 32, 'а': 33, 'п': 34, 'р': 35, 'о': 36, 'л': 37, 'д': 38, 'ж': 39, 'э': 40,
    'я': 44, 'ч': 45, 'с': 46, 'м': 47, 'и': 48, 'т': 49, 'ь': 50, 'б': 51, 'ю': 52, 'ё': 41,
    ' ': 57, '-': 12
}

def press_russian_char(char):
    """Отправляет чистый аппаратный скан-код физической клавиши."""
    low_char = char.lower()

    if low_char in RUS_TO_SCANCODE:
        scancode = RUS_TO_SCANCODE[low_char]
        # Отправляем напрямую числовой скан-код кнопки
        keyboard.send(scancode)
    elif low_char == '.':
        keyboard.send(53)  # Клавиша '/' в русской раскладке дает точку
    elif low_char == ',':
        keyboard.press(42)
        keyboard.send(53)
        keyboard.release(42)

def extract_black_text_mask(bgr_img):
    """Оставляет только черный (еще не напечатанный) текст, убирая зеленый текст и серый фон."""
    b, g, r = cv2.split(bgr_img)
    black_pixels = (r < 95) & (g < 95) & (b < 95)
    binary = np.full(bgr_img.shape[:2], 255, dtype=np.uint8)
    binary[black_pixels] = 0
    return binary

def remove_cursor_box(line_img):
    """
    Точечно стирает рамку курсора ТОЛЬКО вокруг первой буквы (первые 15 пикселей),
    не трогая остальные буквы в строке, чтобы не искажать 'д', 'п', 'в' и т.д.
    """
    res = line_img.copy()
    h, w = res.shape

    # Находим первый столбец с черными пикселями (левая стенка рамки курсора)
    col_sums = np.sum(res == 0, axis=0)
    black_cols = np.where(col_sums > 0)[0]
    if len(black_cols) == 0:
        return res

    x0 = black_cols[0]
    x1 = min(w, x0 + 15)  # Область шириной ровно в один символ с рамкой
    roi = res[:, x0:x1]

    # Находим верхнюю и нижнюю границы рамки курсора в этой узкой зоне
    row_sums = np.sum(roi == 0, axis=1)
    black_rows = np.where(row_sums > 0)[0]
    if len(black_rows) == 0:
        return res

    y_top, y_bot = black_rows[0], black_rows[-1]
    box_h = y_bot - y_top + 1

    # 1. Стираем вертикальные стенки рамки (столбцы, где черные пиксели идут почти на всю высоту рамки)
    roi_col_sums = np.sum(roi == 0, axis=0)
    for cx in range(roi.shape[1]):
        if roi_col_sums[cx] >= box_h - 3:
            roi[:, cx] = 255

    # 2. Стираем верхнюю и нижнюю горизонтальные крышки рамки (по 1-2 пикселя сверху и снизу)
    roi[y_top:min(h, y_top + 2), :] = 255
    roi[max(0, y_bot - 1):y_bot + 1, :] = 255

    res[:, x0:x1] = roi
    return res

def get_complete_lines_images(binary_img, min_line_height=12):
    """
    Возвращает список изображений только ПОЛНЫХ строк.
    Обрезанная снизу строка автоматически отбрасывается.
    """
    h, w = binary_img.shape
    black_pixel_counts = np.sum(binary_img == 0, axis=1)
    lines = []
    in_line = False
    start_y = 0

    for y, count in enumerate(black_pixel_counts):
        if count > 3 and not in_line:
            in_line = True
            start_y = y
        elif count <= 3 and in_line:
            in_line = False
            end_y = y
            if (end_y - start_y) >= min_line_height:
                y1 = max(0, start_y - 3)
                y2 = min(h, end_y + 3)
                lines.append(binary_img[y1:y2, :])
    return lines

def recognize_line(line_img, has_cursor=False):
    """Распознает строку текста строго в нижнем регистре с масштабированием для четкости."""
    img = remove_cursor_box(line_img) if has_cursor else line_img.copy()

    # Увеличиваем изображение в 3 раза и добавляем белые поля для четкости мелкого шрифта
    img_large = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, img_large = cv2.threshold(img_large, 160, 255, cv2.THRESH_BINARY)
    img_padded = cv2.copyMakeBorder(img_large, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)

    # Конфиг без кавычек, чтобы shlex на Windows не выдавал ValueError: No closing quotation
    config = r'--oem 3 --psm 7 -c preserve_interword_spaces=1'

    raw_text = pytesseract.image_to_string(img_padded, lang='rus', config=config)

    # Принудительно переводим все буквы в строчные и оставляем только русские буквы и пробел
    cleaned_text = raw_text.lower()
    cleaned_text = re.sub(r'[^а-яё ]', '', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def type_string_with_speed(text, speed_cpm):
    """Печатает строку с заданной скоростью (знаков в минуту)."""
    base_delay = 60.0 / speed_cpm

    for char in text:
        if keyboard.is_pressed('esc'):
            return False
        press_russian_char(char)
        delay = base_delay * random.uniform(0.88, 1.12)
        time.sleep(delay)

    return True

def main():
    sct = mss.mss()
    print(f"Целевая скорость: {TARGET_SPEED_CPM} знаков/мин (только строчные буквы).")
    print("1. Переключитесь на окно 'СОЛО на клавиатуре 9' (раскладка RU, Caps Lock ВЫКЛ).")
    print("2. Нажмите F8 для старта автопечати (ESC — мгновенная остановка).")

    keyboard.wait('f8')
    print("Старт через 1 секунду...")
    time.sleep(1.0)

    next_line_cache = None

    while True:
        if keyboard.is_pressed('esc'):
            print("\nОстановлено пользователем.")
            break

        sct_img = sct.grab(MONITOR_ROI)
        frame = np.array(sct_img)[:, :, :3]
        binary_mask = extract_black_text_mask(frame)
        line_images = get_complete_lines_images(binary_mask)

        if not line_images:
            print("\nЧерный текст больше не обнаружен (упражнение завершено!).")
            break

        if next_line_cache:
            current_text = next_line_cache
        else:
            current_text = recognize_line(line_images[0], has_cursor=True)

        # Заранее считываем вторую строку, пока на ней еще нет курсора
        if len(line_images) > 1:
            next_line_cache = recognize_line(line_images[1], has_cursor=False)
        else:
            next_line_cache = None

        if not current_text:
            time.sleep(0.1)
            continue

        print(f"Печатаю строку: [{current_text}] (Следующая в буфере: [{next_line_cache}])")

        if not type_string_with_speed(current_text, TARGET_SPEED_CPM):
            print("\nЭкстренная остановка (ESC).")
            break

        # Пробел в конце строки для перехода на следующую
        time.sleep(60.0 / TARGET_SPEED_CPM)
        press_russian_char(' ')

        time.sleep(0.12)

if __name__ == "__main__":
    main()