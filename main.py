from collections import defaultdict
from typing import List, Optional
import re

import numpy as np
from PIL import Image
import streamlit as st

import pytesseract
from pytesseract import TesseractNotFoundError
from deep_translator import GoogleTranslator

_tess = shutil.which("tesseract")
if _tess:
    pytesseract.pytesseract.tesseract_cmd = _tess
else:
    # Common path on Debian-based images used by Streamlit Cloud
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# ----------------- Constants / original config -----------------

split_options = ['L', 'G', 'A', 'LG', 'LGA']
names = {'L': 'Lili', 'G': 'Gergő', 'A': 'Ádi'}

# ----------------- Original helpers (kept as-is or near-identical) -----------------

def translate_item(name: str) -> str:
    # Dutch -> English
    return GoogleTranslator(source='nl', target='en').translate(name)

def extract_receipt_text_from_path(image_path: str) -> str:
    img = Image.open(image_path).convert("RGB")
    return pytesseract.image_to_string(img, lang='nld')

def extract_receipt_text_from_image(img: Image.Image) -> str:
    img = img.convert("RGB")
    return pytesseract.image_to_string(img, lang='nld')

def parse_items(text: str):
    """
    Parsing tuned to handle:
      - Regular item lines with optional 'N x UUU,DD' before the final price
      - Weight lines like '1,20 kg x 2,99 EUR' appended to previous item
      - Discounts: 'In prijs verlaagd', 'Lidl Plus korting', and 'KORTING 25%' etc.
      - Stop at 'Totaal', capture the total
    """
    items = []
    last_item = None
    total_price = None
    lines = text.split('\n')

    # Strict item row: name   [qty x unit]   price   (optional B/C)
    item_pattern = re.compile(
        r'^\s*(.+?)\s+(?:(\d+\s*x\s*\d+,\d{2})\s+)?(-?\d+,\d{2})\s+[BC]?\s*$'
    )

    # Weight continuation (attach to previous item)
    kg_pattern = re.compile(r'^\s*(\d+,\d+\s*kg\s*x\s*\d+,\d{2})\s*EUR\s*$', re.IGNORECASE)

    # Known Dutch discount lines
    price_adjust_pattern = re.compile(r'^\s*(In prijs verlaagd)\s+(-?\d+,\d{2})\s*$', re.IGNORECASE)
    lidl_plus_pattern = re.compile(r'^\s*(Lidl Plus korting)\s+(-?\d+,\d{2})\s*$', re.IGNORECASE)

    # Generic KORTING (e.g., "KORTING 25%   -0,50")
    generic_korting_pattern = re.compile(r'^\s*(KORTING(?:\s*\d+%)*?)\s+(-?\d+,\d{2})\s*$', re.IGNORECASE)

    total_line_pattern = re.compile(r'Totaal\s+(\d+,\d{2})', re.IGNORECASE)

    for i, line in enumerate(lines):
        raw = line.strip()
        if not raw:
            continue

        # Stop at "Totaal"
        if 'Totaal' in raw:
            m = total_line_pattern.search(raw)
            if m:
                total_price = float(m.group(1).replace(',', '.'))
            break

        # Weight continuation attaches to the last item
        m_kg = kg_pattern.match(raw)
        if m_kg and last_item:
            last_item['amount'] = m_kg.group(1)
            continue

        # Discounts (known + generic KORTING)
        for pat in (price_adjust_pattern, lidl_plus_pattern, generic_korting_pattern):
            m_disc = pat.match(raw)
            if m_disc:
                name = m_disc.group(1).strip()
                tr_name = translate_item(name)
                price = float(m_disc.group(2).replace(',', '.'))
                item = {'name': name, 'tr_name': tr_name, 'amount': None, 'price': price}
                items.append(item)
                last_item = item
                break
        else:
            # Regular item row
            m_item = item_pattern.match(raw)
            if m_item:
                name = m_item.group(1).strip()
                amount = m_item.group(2).strip() if m_item.group(2) else None
                price = float(m_item.group(3).replace(',', '.'))
                tr_name = translate_item(name)
                item = {'name': name, 'tr_name': tr_name, 'amount': amount, 'price': price}
                items.append(item)
                last_item = item
            # else: line ignored

    return items, total_price

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

# ----------------- Streamlit UI -----------------

st.set_page_config(page_title="Receipt Splitter", page_icon="🧾", layout="centered")

st.title("🧾 Receipt Bill Splitter")
st.caption("Upload a receipt image → OCR (Tesseract nld) → parse items → assign splits → see balances.")

# Session state init (use names that do NOT collide with dict methods)
if "receipt_items" not in st.session_state:
    st.session_state.receipt_items: List[dict] = []
    st.session_state.total_price: Optional[float] = None
    st.session_state.cur_index: int = 0
    st.session_state.splits: List[str] = []
    st.session_state.payer: Optional[str] = None
    st.session_state.started: bool = False
    st.session_state.ocr_text: Optional[str] = None
    st.session_state.image_preview: Optional[np.ndarray] = None

def reset_state():
    st.session_state.receipt_items = []
    st.session_state.total_price = None
    st.session_state.cur_index = 0
    st.session_state.splits = []
    st.session_state.payer = None
    st.session_state.started = False
    st.session_state.ocr_text = None
    st.session_state.image_preview = None

with st.expander("Upload & OCR", expanded=(len(st.session_state.receipt_items) == 0)):
    file = st.file_uploader("Select digital receipt image (JPG/PNG/BMP)", type=["jpg","jpeg","png","bmp"])
    colA, colB = st.columns([1,1])
    with colA:
        payer_choice = st.radio(
            "Who paid?",
            ['L','G','A'],
            format_func=lambda x: names[x],
            index=0 if st.session_state.payer is None else ['L','G','A'].index(st.session_state.payer)
        )
        st.session_state.payer = payer_choice
    with colB:
        # If your Streamlit version errors on type="primary", just remove that kwarg.
        start_clicked = st.button("Start splitting", type="primary", disabled=(not file))
    if start_clicked and file:
        # OCR – mirror your behavior (Tesseract 'nld')
        image = Image.open(file).convert("RGB")
        st.session_state.image_preview = np.array(image)
        with st.spinner("Scanning receipt with Tesseract (nld)…"):
            text = extract_receipt_text_from_image(image)
        st.session_state.ocr_text = text
        items, total_price = parse_items(text)
        st.session_state.receipt_items = items
        st.session_state.total_price = total_price
        st.session_state.started = True
        st.session_state.cur_index = 0
        st.session_state.splits = []

if st.session_state.image_preview is not None:
    # use_column_width works across Streamlit versions
    st.image(st.session_state.image_preview, caption="Receipt preview", use_container_width=True)

if st.session_state.ocr_text:
    with st.expander("Show OCR text"):
        st.code(st.session_state.ocr_text or "", language="text")

# Splitting workflow (one item at a time)
if st.session_state.started and len(st.session_state.receipt_items) > 0:
    i = st.session_state.cur_index
    total_items = len(st.session_state.receipt_items)

    if i < total_items:
        item = st.session_state.receipt_items[i]
        st.subheader(f"Item {i+1}/{total_items}")
        st.write(f"**Item:** {item['name']}  \n**Translation:** {item['tr_name']}")
        st.write(f"**Amount:** {item['amount'] if item['amount'] else '1'}  \n**Price:** €{item['price']:.2f}")

        cols = st.columns(len(split_options))
        for idx, split in enumerate(split_options):
            if cols[idx].button(split, key=f"split_{i}_{split}"):
                st.session_state.splits.append(split)
                st.session_state.cur_index += 1
                st.rerun()

    # Done → show results (same content as your printout, but formatted)
    if st.session_state.cur_index >= total_items:
        st.success("Splitting complete.")
        st.subheader("Items & splits")

        import pandas as pd
        rows = []
        for it, sp in zip(st.session_state.receipt_items, st.session_state.splits):
            rows.append({
                "Item": it['name'],
                "Translation": it['tr_name'],
                "Amount": it['amount'] if it['amount'] else "",
                "Price (€)": f"{it['price']:.2f}",
                "Split": sp
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        payer_code = st.session_state.payer
        payer_name = names[payer_code]
        total_price = st.session_state.total_price or sum(it['price'] for it in st.session_state.receipt_items)

        st.markdown("---")
        st.subheader("Totals")
        st.write(f"**{payer_name}** paid **€{total_price:.2f}** in total.")

        balances = calculate_balances(st.session_state.receipt_items, st.session_state.splits, payer_code)

        st.subheader("Balances")
        for person, amount in balances.items():
            if person == payer_code:
                continue
            st.write(f"**{names[person]}** pays **{payer_name}** €{amount:.2f}")

        st.markdown("---")
        col1, col2 = st.columns(2)
        if col1.button("Start over"):
            reset_state()
            st.rerun()
        col2.caption("Have a great day!")

else:
    st.info("Upload a receipt, choose who paid, then click **Start splitting**.")
