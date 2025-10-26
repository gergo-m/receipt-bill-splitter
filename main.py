from collections import defaultdict
from typing import List, Optional, Iterable, Tuple, Dict
import re
from itertools import combinations

import numpy as np
from PIL import Image
import streamlit as st

# --- OCR deps / Streamlit Cloud tesseract path ---------------------------
import shutil
import pytesseract
from pytesseract import TesseractNotFoundError
from deep_translator import GoogleTranslator

_tess = shutil.which("tesseract")
if _tess:
    pytesseract.pytesseract.tesseract_cmd = _tess
else:
    # Common path on Debian-based images used by Streamlit Cloud
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# ----------------- Helpers (same behavior as before) ---------------------

def translate_item(name: str) -> str:
    # Dutch -> English
    return GoogleTranslator(source='nl', target='en').translate(name)

def extract_receipt_text_from_image(img: Image.Image) -> str:
    img = img.convert("RGB")
    return pytesseract.image_to_string(img, lang='nld')

# ----------------- Parsing (tuned regexes) -------------------------------

def parse_items(text: str):
    """
    Handles:
      - Regular item lines: name [qty x unit] price (optional B/C)
      - Weight lines: '1,20 kg x 2,99 EUR' appended to previous item
      - Discounts: 'In prijs verlaagd', 'Lidl Plus korting', 'KORTING 25%', etc.
      - Stops at 'Totaal' and captures total
    """
    items = []
    last_item = None
    total_price = None
    lines = text.split('\n')

    item_pattern = re.compile(
        r'^\s*(.+?)\s+(?:(\d+\s*x\s*\d+,\d{2})\s+)?(-?\d+,\d{2})\s+[BC]?\s*$'
    )
    kg_pattern = re.compile(r'^\s*(\d+,\d+\s*kg\s*x\s*\d+,\d{2})\s*EUR\s*$', re.IGNORECASE)
    price_adjust_pattern = re.compile(r'^\s*(In prijs verlaagd)\s+(-?\d+,\d{2})\s*$', re.IGNORECASE)
    lidl_plus_pattern = re.compile(r'^\s*(Lidl Plus korting)\s+(-?\d+,\d{2})\s*$', re.IGNORECASE)
    generic_korting_pattern = re.compile(r'^\s*(KORTING(?:\s*\d+%)*?)\s+(-?\d+,\d{2})\s*$', re.IGNORECASE)
    total_line_pattern = re.compile(r'Totaal\s+(\d+,\d{2})', re.IGNORECASE)

    for line in lines:
        raw = line.strip()
        if not raw:
            continue

        if 'Totaal' in raw:
            m = total_line_pattern.search(raw)
            if m:
                total_price = float(m.group(1).replace(',', '.'))
            break

        m_kg = kg_pattern.match(raw)
        if m_kg and last_item:
            last_item['amount'] = m_kg.group(1)
            continue

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
            m_item = item_pattern.match(raw)
            if m_item:
                name = m_item.group(1).strip()
                amount = m_item.group(2).strip() if m_item.group(2) else None
                price = float(m_item.group(3).replace(',', '.'))
                tr_name = translate_item(name)
                item = {'name': name, 'tr_name': tr_name, 'amount': amount, 'price': price}
                items.append(item)
                last_item = item

    return items, total_price

# ----------------- Dynamic splits (by participant names) -----------------

def initials(name: str) -> str:
    # First non-space character uppercased (keeps accents fine)
    for ch in name.strip():
        if ch.isalpha() or ch.isnumeric():
            return ch.upper()
    return name[:1].upper() if name else "?"

def build_split_options(participants: List[str]) -> List[Dict]:
    """
    Return list of dicts: { label: 'KG', members: ['Kate','George'] }
    Includes all non-empty combinations, ordered by size then alpha label.
    """
    options: List[Dict] = []
    cleaned = [p.strip() for p in participants if p.strip()]
    # de-dup while preserving order
    seen = set()
    ordered = []
    for p in cleaned:
        if p.lower() not in seen:
            ordered.append(p)
            seen.add(p.lower())

    for r in range(1, len(ordered) + 1):
        for combo in combinations(ordered, r):
            label = "".join(initials(n) for n in combo)
            options.append({"label": label, "members": list(combo)})
    # nice order: singles, pairs, ..., within group sort by label
    options.sort(key=lambda o: (len(o["members"]), o["label"]))
    return options

def calculate_balances(items: List[dict], splits: List[Dict], payer: str) -> Dict[str, float]:
    """
    items: list of item dicts with 'price'
    splits: list of {label, members} with 'members' as list of participant names
    payer: participant name who paid the total
    """
    costs = defaultdict(float)
    for item, split in zip(items, splits):
        people = split["members"]
        if not people:
            continue
        per_person = round(item['price'] / len(people), 2)
        for p in people:
            costs[p] += per_person

    balances = {}
    for person, amount in costs.items():
        if person != payer:
            balances[person] = round(amount, 2)
    return balances

# ----------------- Streamlit UI -----------------------------------------

st.set_page_config(page_title="Receipt Splitter", page_icon="🧾", layout="centered")
st.title("🧾 Receipt Bill Splitter")
st.caption("Upload a receipt → OCR (Tesseract ‘nld’) → parse → assign splits → balances. Names and split options are dynamic.")

# Session state init (avoid dict-method name collisions)
if "receipt_items" not in st.session_state:
    st.session_state.receipt_items: List[dict] = []
    st.session_state.total_price: Optional[float] = None
    st.session_state.cur_index: int = 0
    st.session_state.splits: List[Dict] = []         # now: list of {label, members}
    st.session_state.participants: List[str] = ["Kate", "George", "John"]
    st.session_state.split_options: List[Dict] = build_split_options(st.session_state.participants)
    st.session_state.payer: Optional[str] = st.session_state.participants[0]
    st.session_state.started: bool = False
    st.session_state.ocr_text: Optional[str] = None
    st.session_state.image_preview: Optional[np.ndarray] = None

def reset_state(full: bool = False):
    st.session_state.receipt_items = []
    st.session_state.total_price = None
    st.session_state.cur_index = 0
    st.session_state.splits = []
    st.session_state.started = False
    st.session_state.ocr_text = None
    st.session_state.image_preview = None
    if full:
        # keep participants if not a full reset
        st.session_state.participants = ["Kate", "George", "John"]
        st.session_state.split_options = build_split_options(st.session_state.participants)
        st.session_state.payer = st.session_state.participants[0] if st.session_state.participants else None

# --- Participants & OCR upload ------------------------------------------

with st.expander("Participants & Upload", expanded=(len(st.session_state.receipt_items) == 0)):
    # Names input (comma-separated)
    names_input = st.text_input(
        "Participants (comma-separated)",
        value=", ".join(st.session_state.participants),
        help="Example: Kate, George, John"
    )
    # Update participants + split options live
    new_participants = [n.strip() for n in names_input.split(",") if n.strip()]
    if new_participants and new_participants != st.session_state.participants:
        st.session_state.participants = new_participants
        st.session_state.split_options = build_split_options(new_participants)
        if st.session_state.payer not in new_participants:
            st.session_state.payer = new_participants[0]

    colA, colB = st.columns([1, 1])
    with colA:
        payer_choice = st.radio(
            "Who paid?",
            st.session_state.participants,
            index=max(0, st.session_state.participants.index(st.session_state.payer)) if st.session_state.payer in st.session_state.participants else 0
        )
        st.session_state.payer = payer_choice
    with colB:
        file = st.file_uploader("Select receipt image (JPG/PNG/BMP)", type=["jpg", "jpeg", "png", "bmp"])

    start_clicked = st.button("Start splitting", type="primary", disabled=(not file))
    if start_clicked and file:
        image = Image.open(file).convert("RGB")
        st.session_state.image_preview = np.array(image)
        with st.spinner("Scanning receipt with Tesseract (nld)…"):
            try:
                text = extract_receipt_text_from_image(image)
            except TesseractNotFoundError:
                st.error("Tesseract OCR not found. On Streamlit Cloud, add `tesseract-ocr` and `tesseract-ocr-nld` to `packages.txt`, then reboot.")
                st.stop()
        st.session_state.ocr_text = text
        items, total_price = parse_items(text)
        st.session_state.receipt_items = items
        st.session_state.total_price = total_price
        st.session_state.started = True
        st.session_state.cur_index = 0
        st.session_state.splits = []

if st.session_state.image_preview is not None:
    st.image(st.session_state.image_preview, caption="Receipt preview", use_container_width =True)

if st.session_state.ocr_text:
    with st.expander("Show OCR text"):
        st.code(st.session_state.ocr_text or "", language="text")

# --- Splitting workflow (one item at a time) -----------------------------

if st.session_state.started and len(st.session_state.receipt_items) > 0:
    i = st.session_state.cur_index
    total_items = len(st.session_state.receipt_items)

    if i < total_items:
        item = st.session_state.receipt_items[i]
        st.subheader(f"Item {i+1}/{total_items}")
        st.write(f"**Item:** {item['name']}  \n**Translation:** {item['tr_name']}")
        st.write(f"**Amount:** {item['amount'] if item['amount'] else '1'}  \n**Price:** €{item['price']:.2f}")

        # Render dynamic split buttons (singles, pairs, ..., all)
        cols = st.columns(min(6, len(st.session_state.split_options)))  # 6 cols max per row
        for idx, opt in enumerate(st.session_state.split_options):
            c = cols[idx % len(cols)]
            label = opt["label"]
            members = ", ".join(opt["members"])
            if c.button(label, key=f"split_{i}_{label}", help=members):
                st.session_state.splits.append(opt)   # store {label, members}
                st.session_state.cur_index += 1
                st.rerun()

    # Done → results
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
                "Split": sp["label"] + "  (" + ", ".join(sp["members"]) + ")"
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        payer_name = st.session_state.payer
        total_price = st.session_state.total_price or sum(it['price'] for it in st.session_state.receipt_items)

        st.markdown("---")
        st.subheader("Totals")
        st.write(f"**{payer_name}** paid **€{total_price:.2f}** in total.")

        balances = calculate_balances(st.session_state.receipt_items, st.session_state.splits, payer_name)

        st.subheader("Balances")
        # Show everyone except payer (0 or positive amounts to pay the payer)
        for person in st.session_state.participants:
            if person == payer_name:
                continue
            amount = balances.get(person, 0.0)
            st.write(f"**{person}** pays **{payer_name}** €{amount:.2f}")

        st.markdown("---")
        col1, col2 = st.columns(2)
        if col1.button("Start over"):
            reset_state()
            st.rerun()
        if col2.button("Reset everything"):
            reset_state(full=True)
            st.rerun()

else:
    st.info("Add participants, upload a receipt, choose who paid, then click **Start splitting**.")
