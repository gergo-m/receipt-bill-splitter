from PIL import Image
import pytesseract

def extract_receipt_text(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang='nld')
    return text

print(extract_receipt_text("images/2025.08.22_220001778620250822172013.jpg.png"))