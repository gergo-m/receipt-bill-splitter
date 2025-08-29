from PIL import Image
import pytesseract
import re
from deep_translator import GoogleTranslator

def extract_receipt_text(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang='nld')
    return text

def translate_item(name):
    translated = GoogleTranslator(source='nl', target='en').translate(name)
    return translated

def parse_items(text):
    # Matches e.g.: Tomatenblokjes 3 x 0,64 1,92 B
    items = []
    last_item = None
    total_price = None
    lines = text.split('\n')
    pattern = r'(.+?)\s+((?:\d+\s*x\s*\d+,\d+)?)(?:\s+)?(\-?\d+,\d{2})\s+[BC]?'
    price_adjust_pattern = r'(In prijs verlaagd)\s+(\-?\d+,\d{2})'
    for i, line in enumerate(lines):
        # Stop at "Totaal"
        if 'Totaal' in line:
            # Extract total price for reporting
            match = re.search(r'Totaal\s+(\d+,\d{2})', line)
            if match:
                total_price = float(match.group(1).replace(',', '.'))
            break

        # Attach kg lines (next line) to last item if found
        kg_match = re.match(r'(\d+,\d+ kg x \d+,\d+)\s+EUR', line)
        if kg_match and last_item:
            last_item['amount'] = kg_match.group(1)
            continue

        # Match regular item row
        match = re.match(pattern, line)
        if match:
            name = match.group(1).strip()
            amount = match.group(2).replace(',', '.').strip() if match.group(2) else None
            price = float(match.group(3).replace(',', '.'))
            item = {'name': name + ' (' + translate_item(name) + ')', 'amount': amount if amount else None, 'price': price}
            items.append(item)
            last_item = item
            continue

        # Include "In prijs verlaagd" lines
        adj_match = re.match(price_adjust_pattern, line)
        if adj_match:
            name = adj_match.group(1).strip()
            price = float(adj_match.group(2).replace(',', '.'))
            item = {'name': name + ' (' + translate_item(name) + ')', 'amount': None, 'price': price}
            items.append(item)
            continue

    return items, total_price



test_text = extract_receipt_text("images/2025.08.22_220001778620250822172013.jpg.png")
print(test_text)
test_items = parse_items(test_text)
print(test_items)
