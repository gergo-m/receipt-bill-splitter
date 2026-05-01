from collections import defaultdict
from typing import List, Optional, Tuple, Dict
import re
from itertools import combinations

import numpy as np
from PIL import Image
import streamlit as st

# --- OCR deps / Streamlit Cloud tesseract path ---------------------------
import shutil
import pytesseract
from pytesseract import TesseractNotFoundError

_tess = shutil.which("tesseract")
if _tess:
    pytesseract.pytesseract.tesseract_cmd = _tess
else:
    # Common path on Debian-based images used by Streamlit Cloud
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# ----------------- OCR ---------------------------------------------------

def extract_receipt_text_from_image(img: Image.Image) -> str:
    img = img.convert("RGB")
    return pytesseract.image_to_string(img, lang="nld")

# ----------------- Parsing helpers --------------------------------------

def euro_to_float(value: str) -> float:
    """Convert receipt-style euro values such as '1,29' or '-0,57' to float."""
    return float(value.replace(",", "."))


def format_qty(value: Optional[float]) -> str:
    """Human-friendly quantity display without unnecessary trailing zeros."""
    if value is None:
        return "1"
    if float(value).is_integer():
        return str(int(value))
    return (f"{value:.3f}".rstrip("0").rstrip(".")).replace(".", ",")


def make_item(
    name: str,
    price: float,
    quantity: Optional[float] = None,
    unit_price: Optional[float] = None,
    unit_label: Optional[str] = None,
    amount_text: Optional[str] = None,
    is_discount: bool = False,
) -> Dict:
    return {
        "name": name.strip(),
        "quantity": quantity,
        "unit_price": unit_price,
        "unit_label": unit_label,
        "amount_text": amount_text,
        "is_discount": is_discount,
        "price": price,
    }


def item_amount_display(item: Dict) -> str:
    """Display only the amount, not the unit price calculation."""
    if item.get("amount_text"):
        return item["amount_text"]
    if item.get("unit_label") and item.get("quantity") is not None:
        return f"{format_qty(item['quantity'])} {item['unit_label']}"
    return format_qty(item.get("quantity"))


def item_unit_price_display(item: Dict) -> str:
    unit_price = item.get("unit_price")
    if unit_price is None:
        return ""
    unit_label = item.get("unit_label")
    if unit_label:
        return f"€{unit_price:.2f}/{unit_label}"
    return f"€{unit_price:.2f}"

# ----------------- Parsing ----------------------------------------------

def parse_items(text: str) -> Tuple[List[Dict], Optional[float]]:
    """
    Parse Lidl-style receipt OCR text.

    Handles:
      - Regular item lines: name price tax-code
      - Multi-buy item lines: name qty x unit_price total tax-code
        Examples: 'Avocado 3 X 1,29 3,87 B', 'Penne Rigate HWG 2x 0,78 1,56 B'
      - OCR variants such as X, x, Xx, xx, and ×
      - Weight lines following an item: '1,064 kg x 2,98 EUR'
      - Discount/adjustment rows: 'Actieprijs', 'In prijs verlaagd', 'Lidl Plus korting'
      - Stops before checkout/tax/footer sections
    """
    items: List[Dict] = []
    last_product_item: Optional[Dict] = None
    total_price: Optional[float] = None

    item_pattern = re.compile(
        r"^\s*(?P<name>.+?)\s+"
        r"(?:(?P<qty>\d+)\s*[xX×]+\s*(?P<unit_price>\d+,\d{2})\s+)?"
        r"(?P<price>-?\d+,\d{2})\s*[A-Z]?\s*$",
        re.IGNORECASE,
    )

    kg_pattern = re.compile(
        r"^\s*(?P<qty>\d+,\d+)\s*(?P<unit>kg|g)\s*[xX×]+\s*(?P<unit_price>\d+,\d{2})\s*(?:EUR)?\s*$",
        re.IGNORECASE,
    )

    discount_pattern = re.compile(
        r"^\s*(?P<name>Actieprijs|In prijs verlaagd|Lidl Plus korting|KORTING(?:\s*\d+%)?)\s+"
        r"(?P<price>-?\d+,\d{2})\s*$",
        re.IGNORECASE,
    )

    total_line_pattern = re.compile(r"^\s*Totaal\s+(?P<total>\d+,\d{2})\s*$", re.IGNORECASE)

    footer_start_pattern = re.compile(
        r"^\s*(Aantal\b|Bankpas\b|Kopie Kaarthouder\b|Terminal\b|AID\b|DEBIT\b|Kaart\b|Volgnr\b|Betaling\b|%\b|Waarvan\b|DANK U WEL\b|Kortingscoupons\b|Aankoop gedaan bij\b)",
        re.IGNORECASE,
    )

    for raw_line in text.splitlines():
        raw = raw_line.strip()
        if not raw:
            continue

        m_total = total_line_pattern.match(raw)
        if m_total:
            total_price = euro_to_float(m_total.group("total"))
            break

        if footer_start_pattern.match(raw):
            continue

        m_kg = kg_pattern.match(raw)
        if m_kg and last_product_item:
            qty_text = m_kg.group("qty")
            unit = m_kg.group("unit").lower()
            unit_price_text = m_kg.group("unit_price")
            last_product_item["quantity"] = euro_to_float(qty_text)
            last_product_item["unit_price"] = euro_to_float(unit_price_text)
            last_product_item["unit_label"] = unit
            last_product_item["amount_text"] = f"{qty_text} {unit}"
            continue

        m_disc = discount_pattern.match(raw)
        if m_disc:
            name = m_disc.group("name").strip()
            price = euro_to_float(m_disc.group("price"))
            items.append(make_item(name=name, price=price, is_discount=True))
            continue

        m_item = item_pattern.match(raw)
        if m_item:
            name = m_item.group("name").strip()
            price = euro_to_float(m_item.group("price"))
            qty = euro_to_float(m_item.group("qty")) if m_item.group("qty") else None
            unit_price = euro_to_float(m_item.group("unit_price")) if m_item.group("unit_price") else None

            item = make_item(
                name=name,
                quantity=qty,
                unit_price=unit_price,
                price=price,
            )
            items.append(item)
            last_product_item = item

    return items, total_price

# ----------------- Dynamic splits (by participant names) -----------------

def initials(name: str) -> str:
    for ch in name.strip():
        if ch.isalpha() or ch.isnumeric():
            return ch.upper()
    return name[:1].upper() if name else "?"


def build_split_options(participants: List[str]) -> List[Dict]:
    options: List[Dict] = []
    cleaned = [p.strip() for p in participants if p.strip()]

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

    options.sort(key=lambda o: (len(o["members"]), o["label"]))
    return options


def calculate_balances(items: List[dict], splits: List[Dict], payer: str) -> Dict[str, float]:
    costs = defaultdict(float)
    for item, split in zip(items, splits):
        people = split["members"]
        if not people:
            continue
        per_person = round(item["price"] / len(people), 2)
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

if "receipt_items" not in st.session_state:
    st.session_state.receipt_items: List[dict] = []
    st.session_state.total_price: Optional[float] = None
    st.session_state.cur_index: int = 0
    st.session_state.splits: List[Dict] = []
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
        st.session_state.participants = ["Kate", "George", "John"]
        st.session_state.split_options = build_split_options(st.session_state.participants)
        st.session_state.payer = st.session_state.participants[0] if st.session_state.participants else None

# --- Participants & OCR upload ------------------------------------------

with st.expander("Participants & Upload", expanded=(len(st.session_state.receipt_items) == 0)):
    names_input = st.text_input(
        "Participants (comma-separated)",
        value=", ".join(st.session_state.participants),
        help="Example: Kate, George, John",
    )

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
            index=max(0, st.session_state.participants.index(st.session_state.payer))
            if st.session_state.payer in st.session_state.participants
            else 0,
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
    st.image(st.session_state.image_preview, caption="Receipt preview", use_container_width=True)

if st.session_state.ocr_text:
    with st.expander("Show OCR text"):
        st.code(st.session_state.ocr_text or "", language="text")

# --- Splitting workflow --------------------------------------------------

if st.session_state.started and len(st.session_state.receipt_items) > 0:
    i = st.session_state.cur_index
    total_items = len(st.session_state.receipt_items)

    if i < total_items:
        item = st.session_state.receipt_items[i]
        st.subheader(f"Item {i + 1}/{total_items}")

        st.write(f"**Item:** {item['name']}")
        st.write(f"**Amount:** {item_amount_display(item)}")
        if item_unit_price_display(item):
            st.write(f"**Unit price:** {item_unit_price_display(item)}")
        st.write(f"**Price:** €{item['price']:.2f}")

        cols = st.columns(min(6, len(st.session_state.split_options)))
        for idx, opt in enumerate(st.session_state.split_options):
            c = cols[idx % len(cols)]
            label = opt["label"]
            members = ", ".join(opt["members"])
            if c.button(label, key=f"split_{i}_{label}", help=members):
                st.session_state.splits.append(opt)
                st.session_state.cur_index += 1
                st.rerun()

    if st.session_state.cur_index >= total_items:
        st.success("Splitting complete.")
        st.subheader("Items & splits")

        import pandas as pd
        rows = []
        for it, sp in zip(st.session_state.receipt_items, st.session_state.splits):
            rows.append({
                "Item": it["name"],
                "Amount": item_amount_display(it),
                "Unit price": item_unit_price_display(it),
                "Price (€)": f"{it['price']:.2f}",
                "Split": sp["label"] + "  (" + ", ".join(sp["members"]) + ")",
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        payer_name = st.session_state.payer
        total_price = st.session_state.total_price or sum(it["price"] for it in st.session_state.receipt_items)

        st.markdown("---")
        st.subheader("Totals")
        st.write(f"**{payer_name}** paid **€{total_price:.2f}** in total.")

        balances = calculate_balances(st.session_state.receipt_items, st.session_state.splits, payer_name)

        st.subheader("Balances")
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