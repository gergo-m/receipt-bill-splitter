from collections import defaultdict

from PIL import Image
import pytesseract
import re
from deep_translator import GoogleTranslator
import tkinter as tk
from tkinter import filedialog, messagebox
import threading

split_options = ['L', 'G', 'A', 'LG', 'LGA']
names = {'L': 'Lili', 'G': 'Gergő', 'A': 'Ádi'}

def extract_receipt_text(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang='nld')
    return text

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
            tr_name = translate_item(name)
            amount = match.group(2).replace(',', '.').strip() if match.group(2) else None
            price = float(match.group(3).replace(',', '.'))
            item = {'name': name, 'tr_name': tr_name, 'amount': amount if amount else None, 'price': price}
            items.append(item)
            last_item = item
            continue

        # Include "In prijs verlaagd" lines
        adj_match = re.match(price_adjust_pattern, line)
        if adj_match:
            name = adj_match.group(1).strip()
            tr_name = translate_item(name)
            price = float(adj_match.group(2).replace(',', '.'))
            item = {'name': name, 'tr_name': tr_name, 'amount': None, 'price': price}
            items.append(item)
            continue

    return items, total_price

def translate_item(name):
    translated = GoogleTranslator(source='nl', target='en').translate(name)
    return translated

def calculate_balances(items, splits, payer):
    person_map = {'L': ['L'], 'G': ['G'], 'A': ['A'], 'LG': ['L', 'G'], 'LGA': ['L', 'G', 'A']}
    costs = defaultdict(float)
    for item, split in zip(items, splits):
        people = person_map[split]
        per_person = round(item['price']/len(people), 2)
        for p in people:
            costs[p] += per_person
    balances = {}
    for person in person_map['LGA']:
        if person != payer:
            balances[person] = round(costs[person], 2)
    return balances

class ReceiptSplitter(tk.Toplevel):
    def __init__(self, items, total_price):
        super().__init__()
        self.title("Receipt Splitter")
        self.items = items
        self.total_price = total_price
        self.cur_index = 0
        self.splits = []
        self.payer = tk.StringVar()
        tk.Label(self, text="Who paid?").pack()
        for p in ['L', 'G', 'A']:
            tk.Radiobutton(self, text=names[p], variable=self.payer, value=p).pack(anchor='w')
        tk.Button(self, text="Start splitting", command=self.start_splitting).pack(pady=10)

    def start_splitting(self):
        if not self.payer.get():
            tk.messagebox.showerror('Who paid?', 'Please select payer before proceeding.')
            return
        self.next_item()

    def next_item(self):
        if self.cur_index >= len(self.items):
            self.show_results()
            self.destroy()
            return
        if hasattr(self, 'item_frame'):
            self.item_frame.destroy()
        self.item_frame = tk.Frame(self)
        self.item_frame.pack()
        item = self.items[self.cur_index]
        tk.Label(self.item_frame, text=f"Item: {item['name']} ({item['tr_name']})").pack()
        tk.Label(self.item_frame, text=f"Amount: {item['amount'] if item['amount'] else 1}\nPrice: €{item['price']:.2f}").pack()
        for split in split_options:
            btn = tk.Button(self.item_frame, text=split, command=lambda s=split: self.choose_split(s))
            btn.pack(side='left')

    def choose_split(self, split):
        self.splits.append(split)
        self.item_frame.destroy()
        self.cur_index += 1
        self.next_item()

    def show_results(self):
        print('-------------------------------------')
        print(f"{'Item':35} {'Translation':30} {'Amount':15} {'Price':8} {'Split':5}")
        print('-' * 100)
        for item, split in zip(self.items, self.splits):
            name_str = f"{item['name'][:33]:35}"  # max length 33 + 2 spaces padding
            translation_str = f"{item['tr_name'][:28]:30}"  # max 28 + 2 spaces
            amount_str = f"{item['amount'] if item['amount'] else '':15}"
            price_str = f"{item['price']:.2f}"
            split_str = f"({split})"
            print(f"{name_str} {translation_str} {amount_str} {price_str:>8} {split_str:5}")
        print('-' * 100)
        print(f"{names[self.payer.get()]} paid {self.total_price:.2f} EUR in total")
        splits = calculate_balances(self.items, self.splits, self.payer.get())
        for person, amount in splits.items():
            if person == self.payer.get():
                continue
            print(f"{names[person]} pays {names[self.payer.get()]} {amount:.2f} EUR")
        exit(0)

class ReceiptApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Upload Receipt")
        self.geometry("300x100")
        self.btn = tk.Button(self, text="Select Digital Receipt Image", command=self.upload_file)
        self.btn.pack(expand=True)
        self.loading_popup = None

    def show_loading(self):
        self.loading_popup = tk.Toplevel(self)
        self.loading_popup.title("Please wait")
        self.loading_popup.geometry("200x80")
        tk.Label(self.loading_popup, text="Scanning receipt...\nPlease wait.").pack(pady=20)
        self.loading_popup.grab_set()
        self.update()

    def close_loading(self):
        if self.loading_popup:
            self.loading_popup.destroy()
            self.loading_popup = None

    def process_receipt(self, file_path):
        try:
            receipt_text = extract_receipt_text(file_path)
            receipt_items, total_price = parse_items(receipt_text)
            # Close loading popup in main thread
            self.after(0, self.close_loading)
            # Close file select window and open splitter
            self.after(0, lambda: self.open_splitter(receipt_items, total_price))  # close main window
        except Exception as e:
            self.after(0, self.close_loading)
            self.after(0, lambda: messagebox.showerror("Error", f"Failed to process image: \n{e}"))

    def open_splitter(self, receipt_items, total_price):
        self.withdraw()
        splitter = ReceiptSplitter(receipt_items, total_price)
        splitter.grab_set()

    def upload_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Digital Receipt Image",
            filetypes=(("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*"))
        )
        if file_path:
            self.show_loading()
            # Run the OCR and parsing in a separate thread
            threading.Thread(target=self.process_receipt, args=(file_path,), daemon=True).start()


if __name__ == '__main__':
    app = ReceiptApp()
    app.mainloop()
