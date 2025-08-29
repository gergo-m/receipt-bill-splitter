from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.properties import StringProperty, ListProperty, NumericProperty
from kivy.clock import Clock, mainthread
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserIconView
from threading import Thread
from collections import defaultdict
from PIL import Image
import pytesseract
import re
from deep_translator import GoogleTranslator

split_options = ['L', 'G', 'A', 'LG', 'LGA']
names = {'L': 'Lili', 'G': 'Gergő', 'A': 'Ádi'}

def extract_receipt_text(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang='nld')
    return text

def parse_items(text):
    items = []
    last_item = None
    total_price = None
    lines = text.split('\n')
    pattern = r'(.+?)\s+((?:\d+\s*x\s*\d+,\d+)?)(?:\s+)?(\-?\d+,\d{2})\s+[BC]?'
    price_adjust_pattern = r'(In prijs verlaagd)\s+(\-?\d+,\d{2})'
    for i, line in enumerate(lines):
        if 'Totaal' in line:
            match = re.search(r'Totaal\s+(\d+,\d{2})', line)
            if match:
                total_price = float(match.group(1).replace(',', '.'))
            break
        kg_match = re.match(r'(\d+,\d+ kg x \d+,\d+)\s+EUR', line)
        if kg_match and last_item:
            last_item['amount'] = kg_match.group(1)
            continue
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
    try:
        return GoogleTranslator(source='nl', target='en').translate(name)
    except Exception:
        return name

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

class LoadDialog(BoxLayout):
    load = None  # callback for selecting file
    cancel = None

class ReceiptSplitterLayout(BoxLayout):
    payer = StringProperty('')
    split_choices = ListProperty([])
    items = ListProperty([])
    total_price = NumericProperty(0)
    cur_index = NumericProperty(0)

    def start_splitting(self):
        if not self.payer:
            self.show_popup("Please select who paid before proceeding.")
            return
        self.cur_index = 0
        self.split_choices = []
        self.show_next_item()

    def show_next_item(self):
        if self.cur_index >= len(self.items):
            self.show_results()
            return
        self.ids.item_label.text = f"Item: {self.items[self.cur_index]['name']} ({self.items[self.cur_index]['tr_name']})"
        amt = self.items[self.cur_index]['amount']
        price = self.items[self.cur_index]['price']
        self.ids.details_label.text = f"Amount: {amt if amt else ''}\nPrice: €{price:.2f}"
        self.ids.buttons_box.clear_widgets()
        for split in split_options:
            btn = Button(text=split, on_press=lambda btn, s=split: self.choose_split(s))
            self.ids.buttons_box.add_widget(btn)

    def choose_split(self, split):
        self.split_choices.append(split)
        self.cur_index +=1
        self.show_next_item()

    def show_results(self):
        result_lines = []
        result_lines.append(f"{names[self.payer]} paid €{self.total_price:.2f} in total\nSplits:\n")
        splits = calculate_balances(self.items, self.split_choices, self.payer)
        for person, amount in splits.items():
            result_lines.append(f"{names[person]} pays {names[self.payer]} €{amount:.2f}")
        result_lines.append("\nFull breakdown:")
        for item, split in zip(self.items, self.split_choices):
            amt = item['amount'] if item['amount'] else ''
            result_lines.append(f"{item['name']} ({item['tr_name']})  {amt}  €{item['price']:.2f}  ({split})")
        self.show_popup("\n".join(result_lines))

    def show_popup(self, text):
        popup = Popup(title='Result', content=Label(text=text), size_hint=(.9, .9))
        popup.open()

class ReceiptApp(App):
    def build(self):
        self.title = "Receipt Splitter"
        self.root = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        return self.root

    def dismiss_popup(self):
        self.root_popup.dismiss()

    def load(self, path, filename):
        if filename:
            self.root_popup.dismiss()
            filepath = filename[0]
            self.start_receipt_processing(filepath)

    def start_receipt_processing(self, filepath):
        self.root_popup = Popup(title='Scanning receipt...', content=Label(text='Please wait...'), size_hint=(.5,.5))
        self.root_popup.open()
        Thread(target=self.process_receipt, args=(filepath,), daemon=True).start()

    def process_receipt(self, filepath):
        try:
            text = extract_receipt_text(filepath)
            items, total_price = parse_items(text)
            self.close_popup()
            # Switch to splitter UI on main thread
            Clock.schedule_once(lambda dt: self.show_splitter(items, total_price), 0)
        except Exception as e:
            self.close_popup()
            Clock.schedule_once(lambda dt: self.show_error(str(e)), 0)

    @mainthread
    def close_popup(self):
        if self.root_popup:
            self.root_popup.dismiss()
            self.root_popup = None

    def show_splitter(self, items, total_price):
        splitter = ReceiptSplitterLayout()
        splitter.items = items
        splitter.total_price = total_price
        self.root.clear_widgets()
        self.root.add_widget(splitter)

    @mainthread
    def show_error(self, msg):
        popup = Popup(title='Error', content=Label(text=msg), size_hint=(.9,.9))
        popup.open()

if __name__ == '__main__':
    from threading import Thread
    from kivy.clock import Clock, mainthread
    ReceiptApp().run()
