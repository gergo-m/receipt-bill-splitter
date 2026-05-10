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

def prepare_ocr_image(img: Image.Image) -> Image.Image:
    """Return a larger, high-contrast image that Tesseract handles better."""
    img = img.convert("L")

    # Upscale small/mobile screenshots. Tesseract is much happier around 2x-3x.
    w, h = img.size
    if w < 1200:
        scale = max(2, int(round(1200 / max(w, 1))))
        img = img.resize((w * scale, h * scale), Image.Resampling.LANCZOS)

    # Simple thresholding keeps receipt text crisp without extra dependencies.
    img = img.point(lambda px: 255 if px > 185 else 0)
    return img


def extract_receipt_text_from_image(img: Image.Image, config: str = "--psm 4") -> str:
    """OCR one receipt image using a configurable Tesseract page segmentation mode."""
    prepared = prepare_ocr_image(img)
    return pytesseract.image_to_string(prepared, lang="nld", config=config)


def ocr_text_candidates(img: Image.Image) -> List[Tuple[str, str]]:
    """
    Return OCR candidates using modes that work differently on Lidl receipts.

    Lidl receipts sometimes OCR as rows and sometimes as two columns. Trying a few
    page segmentation modes and then letting the parser choose the best result is
    much more stable than trusting one Tesseract layout guess.
    """
    configs = [
        "--psm 4",   # single column-ish receipt, usually best for normal rows
        "--psm 3",   # fully automatic page segmentation
        "--psm 11",  # sparse text; often exposes separated columns cleanly
        "--psm 6",   # uniform block; useful fallback on cropped receipts
    ]

    out: List[Tuple[str, str]] = []
    seen = set()

    for cfg in configs:
        try:
            txt = extract_receipt_text_from_image(img, config=cfg)
        except Exception:
            continue

        key = re.sub(r"\s+", " ", txt).strip()[:1000]
        if txt.strip() and key not in seen:
            out.append((cfg, txt))
            seen.add(key)

    return out

# ----------------- Parsing helpers --------------------------------------

EURO_RE = r"-?\d+[,.]\d{2}"


def euro_to_float(value: str) -> float:
    """Convert receipt-style euro values such as '1,29', '1.29', or '-0,57' to float."""
    return float(value.strip().replace("€", "").replace(",", "."))


def format_qty(value: Optional[float]) -> str:
    """Human-friendly quantity display without unnecessary trailing zeros."""
    if value is None:
        return "1"
    if float(value).is_integer():
        return str(int(value))
    return (f"{value:.3f}".rstrip("0").rstrip(".")).replace(".", ",")


def clean_name(name: str) -> str:
    """Remove OCR junk around item names while preserving useful product text."""
    name = re.sub(r"\s+", " ", name or "").strip()
    name = re.sub(r"\b[BC]$", "", name).strip()  # trailing VAT code if OCR glued it to a name
    return name


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
        "name": clean_name(name),
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

# ----------------- Lidl parsing -----------------------------------------

DISCOUNT_LABEL_RE = re.compile(
    r"^(Actieprijs|In prijs verlaagd|Lidl Plus korting|KORTING(?:\s*\d+%)?)$",
    re.IGNORECASE,
)

MULTIBUY_AT_END_RE = re.compile(
    rf"^(?P<name>.+?)\s+(?P<qty>\d+)\s*[xX×]+\s*(?P<unit_price>{EURO_RE})$",
    re.IGNORECASE,
)

WEIGHT_LINE_RE = re.compile(
    rf"^(?P<qty>\d+[,.]\d+)\s*(?P<unit>kg|g)\s*[xX×]+\s*(?P<unit_price>{EURO_RE})\s*(?:EUR)?$",
    re.IGNORECASE,
)

INLINE_ITEM_RE = re.compile(
    rf"^\s*(?P<name>.+?)\s+"
    rf"(?:(?P<qty>\d+)\s*[xX×]+\s*(?P<unit_price>{EURO_RE})\s+)?"
    rf"(?P<price>{EURO_RE})\s*[A-Z]?\s*$",
    re.IGNORECASE,
)

INLINE_DISCOUNT_RE = re.compile(
    rf"^\s*(?P<name>Actieprijs|In prijs verlaagd|Lidl Plus korting|KORTING(?:\s*\d+%)?)\s+"
    rf"(?P<price>{EURO_RE})\s*$",
    re.IGNORECASE,
)

TOTAL_RE = re.compile(rf"\bTotaal\b(?:\s+EUR)?\s+(?P<total>{EURO_RE})\b", re.IGNORECASE)

STOP_RE = re.compile(
    r"^(Aantal\b|Totaal\b|Bankpas\b|Kopie Kaarthouder\b|Terminal\b|AID\b|DEBIT\b|Kaart\b|Volgnr\b|Kaartbetaling\b|geen\s+CVM\b|Betaling\b|%\b|Waarvan\b|DANK U WEL\b|Bespaar\b|Bedankt\b|Kortingscoupons\b|Punten\b|Aankoop gedaan bij\b|Filiaal informatie\b)",
    re.IGNORECASE,
)

HEADER_OR_NOISE_RE = re.compile(
    r"^(Kopie kassabon|L[0-9I]?DL|Lidl\b|Urkhovenseweg\b|\d{4}\s+[A-Z]{2}\b|OMSCHRIJVING\b|EUR\b)$",
    re.IGNORECASE,
)

TAX_CODE_NOISE_RE = re.compile(r"^[A-Z\s=]{1,20}$")


def is_discount_label(line: str) -> bool:
    return bool(DISCOUNT_LABEL_RE.match(line.strip()))


def parse_multibuy_name(line: str) -> Tuple[str, Optional[float], Optional[float]]:
    """
    Parse product names that include quantity/unit price but not final total.

    Example:
      'Hummus 2 Xx 1,38' -> ('Hummus', 2, 1.38)
    """
    m = MULTIBUY_AT_END_RE.match(line.strip())
    if not m:
        return clean_name(line), None, None

    return (
        clean_name(m.group("name")),
        float(m.group("qty")),
        euro_to_float(m.group("unit_price")),
    )


def extract_total(text: str) -> Optional[float]:
    for line in text.splitlines():
        m = TOTAL_RE.search(line.strip())
        if m:
            return euro_to_float(m.group("total"))
    return None


def receipt_body_lines(text: str) -> List[str]:
    """Return normalized non-empty lines from the receipt item area."""
    lines = [re.sub(r"\s+", " ", ln.strip()) for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]

    start = 0
    for idx, line in enumerate(lines):
        if re.match(r"^OMSCHRIJVING\b", line, re.IGNORECASE):
            start = idx + 1
            break

    return lines[start:]


def price_tokens_from_lines(lines: List[str]) -> List[float]:
    """Extract one price per line from the right-hand price column."""
    prices: List[float] = []

    for line in lines:
        s = line.strip()

        if STOP_RE.match(s):
            break

        # A Lidl price-column line is usually exactly '9,25 B' or '-0,06'.
        m = re.match(rf"^(?P<price>{EURO_RE})\s*[BC]?\s*$", s, re.IGNORECASE)
        if m:
            prices.append(euro_to_float(m.group("price")))

    return prices


def descriptor_from_line(line: str) -> Optional[Dict]:
    """Convert a product/discount description line into an item descriptor without price."""
    s = line.strip()

    if not s:
        return None
    if HEADER_OR_NOISE_RE.match(s):
        return None
    if STOP_RE.match(s):
        return None

    # The new Lidl OCR sometimes emits VAT letters as a separate vertical column.
    if TAX_CODE_NOISE_RE.match(s) and not any(ch.islower() for ch in s):
        return None

    if is_discount_label(s):
        return {
            "name": clean_name(s),
            "quantity": None,
            "unit_price": None,
            "unit_label": None,
            "amount_text": None,
            "is_discount": True,
        }

    name, qty, unit_price = parse_multibuy_name(s)
    if not name:
        return None

    return {
        "name": name,
        "quantity": qty,
        "unit_price": unit_price,
        "unit_label": None,
        "amount_text": None,
        "is_discount": False,
    }


def parse_column_receipt(text: str) -> Tuple[List[Dict], Optional[float]]:
    """
    Parse Lidl OCR where Tesseract read the receipt as columns:
      product names first, then the EUR price column.

    This started happening on the newer Lidl layout. The content is still there,
    but item names and item prices are no longer on the same OCR line.
    """
    body = receipt_body_lines(text)
    total_price = extract_total(text)

    # Find the standalone EUR header that starts the price column.
    eur_idx = None
    for idx, line in enumerate(body):
        if re.fullmatch(r"EUR", line.strip(), re.IGNORECASE):
            eur_idx = idx
            break

    if eur_idx is None:
        return [], total_price

    name_lines = body[:eur_idx]
    price_lines = body[eur_idx + 1:]
    prices = price_tokens_from_lines(price_lines)

    descriptors: List[Dict] = []
    last_product_desc: Optional[Dict] = None

    for line in name_lines:
        s = line.strip()

        if not s:
            continue

        m_weight = WEIGHT_LINE_RE.match(s)
        if m_weight and last_product_desc is not None:
            qty_text = m_weight.group("qty")
            unit = m_weight.group("unit").lower()
            unit_price = m_weight.group("unit_price")

            last_product_desc["quantity"] = euro_to_float(qty_text)
            last_product_desc["unit_price"] = euro_to_float(unit_price)
            last_product_desc["unit_label"] = unit
            last_product_desc["amount_text"] = f"{qty_text} {unit}"
            continue

        desc = descriptor_from_line(s)
        if desc is None:
            continue

        descriptors.append(desc)
        if not desc.get("is_discount"):
            last_product_desc = desc

    # Pair each item/discount description with the corresponding price in order.
    # If OCR drops a price, keep the reliable pairs rather than inventing data.
    items: List[Dict] = []
    for desc, price in zip(descriptors, prices):
        items.append(
            make_item(
                name=desc["name"],
                price=price,
                quantity=desc.get("quantity"),
                unit_price=desc.get("unit_price"),
                unit_label=desc.get("unit_label"),
                amount_text=desc.get("amount_text"),
                is_discount=desc.get("is_discount", False),
            )
        )

    # Only trust this mode when it found a meaningful receipt-sized list.
    if len(items) < 5 or len(prices) < max(5, int(len(descriptors) * 0.5)):
        return [], total_price

    return items, total_price


def parse_inline_receipt(text: str) -> Tuple[List[Dict], Optional[float]]:
    """Parse the older/easier OCR where product and price are on the same line."""
    items: List[Dict] = []
    last_product_item: Optional[Dict] = None
    total_price = extract_total(text)

    for raw_line in receipt_body_lines(text):
        raw = raw_line.strip()

        if not raw:
            continue
        if TOTAL_RE.search(raw):
            break
        if STOP_RE.match(raw):
            continue
        if HEADER_OR_NOISE_RE.match(raw):
            continue

        m_weight = WEIGHT_LINE_RE.match(raw)
        if m_weight and last_product_item:
            qty_text = m_weight.group("qty")
            unit = m_weight.group("unit").lower()
            unit_price_text = m_weight.group("unit_price")

            last_product_item["quantity"] = euro_to_float(qty_text)
            last_product_item["unit_price"] = euro_to_float(unit_price_text)
            last_product_item["unit_label"] = unit
            last_product_item["amount_text"] = f"{qty_text} {unit}"
            continue

        m_disc = INLINE_DISCOUNT_RE.match(raw)
        if m_disc:
            items.append(
                make_item(
                    name=m_disc.group("name"),
                    price=euro_to_float(m_disc.group("price")),
                    is_discount=True,
                )
            )
            continue

        m_item = INLINE_ITEM_RE.match(raw)
        if m_item:
            name = clean_name(m_item.group("name"))
            if HEADER_OR_NOISE_RE.match(name) or name.lower().startswith("totaal"):
                continue

            qty = float(m_item.group("qty")) if m_item.group("qty") else None
            unit_price = euro_to_float(m_item.group("unit_price")) if m_item.group("unit_price") else None
            price = euro_to_float(m_item.group("price"))

            item = make_item(
                name=name,
                quantity=qty,
                unit_price=unit_price,
                price=price,
            )
            items.append(item)
            last_product_item = item

    return items, total_price


def parse_items(text: str) -> Tuple[List[Dict], Optional[float]]:
    """
    Parse Lidl-style receipt OCR.

    The important fix is that this parser handles both OCR reading orders:
      - normal row order: 'Avocado 3 X 1,29 3,87 B'
      - new column order: product names first, then a separate EUR price column

    The new Lidl receipt layout, Lidl Plus points/coupons, and the long digital receipt
    area make Tesseract more likely to return column-order OCR. A line-by-line parser
    then sees only names without prices and later a pile of prices without names. That
    is why the app suddenly started finding only 'Totaal EUR' or a few wrong large rows.
    """
    column_items, column_total = parse_column_receipt(text)
    inline_items, inline_total = parse_inline_receipt(text)

    # Choose the mode that found more plausible item rows. Column mode wins ties because
    # the broken newer OCR otherwise creates a few bogus inline matches from the footer.
    if len(column_items) >= len(inline_items) and len(column_items) > 0:
        return column_items, column_total or inline_total

    return inline_items, inline_total or column_total


def parse_quality_score(items: List[Dict], total_price: Optional[float]) -> float:
    """Score parsed OCR results so we can pick the best Tesseract candidate."""
    if not items:
        return -10_000.0

    item_sum = sum(item["price"] for item in items)
    score = float(len(items)) * 10.0

    # Prefer parses whose rows add up to the printed total, but do not reject a
    # parse completely because OCR may drop one price line.
    if total_price is not None:
        diff = abs(item_sum - total_price)
        score -= min(diff, 25.0) * 8.0

        if diff < 0.03:
            score += 250.0
        elif diff < 1.00:
            score += 100.0

    # Discounts are expected on Lidl Plus receipts. A parse with no discounts is
    # often a footer/total false-positive parse.
    score += sum(1 for item in items if item.get("is_discount")) * 2.0

    return score


def scan_and_parse_receipt(img: Image.Image) -> Tuple[str, List[Dict], Optional[float], str]:
    """OCR with multiple layouts, parse each result, and keep the best one."""
    best_text = ""
    best_items: List[Dict] = []
    best_total: Optional[float] = None
    best_cfg = ""
    best_score = -10_000.0

    for cfg, text in ocr_text_candidates(img):
        items, total = parse_items(text)
        score = parse_quality_score(items, total)

        if score > best_score:
            best_text = text
            best_items = items
            best_total = total
            best_cfg = cfg
            best_score = score

    return best_text, best_items, best_total, best_cfg

# ----------------- Dynamic splits by participant names -------------------

def initials(name: str) -> str:
    for ch in name.strip():
        if ch.isalpha() or ch.isnumeric():
            return ch.upper()
    return name[:1].upper() if name else "?"


def build_split_options(participants: List[str]) -> List[Dict]:
    """Return list of dicts: { label: 'KG', members: ['Kate','George'] }."""
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

        per_person = item["price"] / len(people)
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
st.caption("Upload a receipt → OCR → parse Lidl rows → assign splits → balances.")

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
    st.session_state.ocr_config: Optional[str] = None
    st.session_state.image_preview: Optional[np.ndarray] = None


def reset_state(full: bool = False):
    st.session_state.receipt_items = []
    st.session_state.total_price = None
    st.session_state.cur_index = 0
    st.session_state.splits = []
    st.session_state.started = False
    st.session_state.ocr_text = None
    st.session_state.ocr_config = None
    st.session_state.image_preview = None

    if full:
        st.session_state.participants = ["Kate", "George", "John"]
        st.session_state.split_options = build_split_options(st.session_state.participants)
        st.session_state.payer = st.session_state.participants[0]

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
        file = st.file_uploader(
            "Select receipt image (JPG/PNG/BMP)",
            type=["jpg", "jpeg", "png", "bmp"],
        )

    start_clicked = st.button("Start splitting", type="primary", disabled=(not file))

    if start_clicked and file:
        image = Image.open(file).convert("RGB")
        st.session_state.image_preview = np.array(image)

        with st.spinner("Scanning receipt with Tesseract…"):
            try:
                text, items, total_price, ocr_config = scan_and_parse_receipt(image)
            except TesseractNotFoundError:
                st.error(
                    "Tesseract OCR not found. On Streamlit Cloud, add `tesseract-ocr` "
                    "and `tesseract-ocr-nld` to `packages.txt`, then reboot."
                )
                st.stop()

        st.session_state.ocr_text = text
        st.session_state.ocr_config = ocr_config
        st.session_state.receipt_items = items
        st.session_state.total_price = total_price
        st.session_state.started = True
        st.session_state.cur_index = 0
        st.session_state.splits = []

if st.session_state.image_preview is not None:
    st.image(st.session_state.image_preview, caption="Receipt preview", use_container_width=True)

if st.session_state.ocr_text:
    with st.expander("Show OCR text"):
        if st.session_state.get("ocr_config"):
            st.caption(f"Selected OCR mode: {st.session_state.ocr_config}")
        st.code(st.session_state.ocr_text or "", language="text")

    with st.expander("Parsed items debug"):
        parsed_total = sum(item["price"] for item in st.session_state.receipt_items)
        st.write(f"Parsed rows: **{len(st.session_state.receipt_items)}**")
        st.write(f"Parsed item sum: **€{parsed_total:.2f}**")

        if st.session_state.total_price is not None:
            st.write(f"Receipt total: **€{st.session_state.total_price:.2f}**")
            st.write(f"Difference: **€{parsed_total - st.session_state.total_price:.2f}**")

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
        receipt_total = st.session_state.total_price
        parsed_total = sum(it["price"] for it in st.session_state.receipt_items)
        total_to_show = receipt_total if receipt_total is not None else parsed_total

        st.markdown("---")
        st.subheader("Totals")
        st.write(f"**{payer_name}** paid **€{total_to_show:.2f}** in total.")
        st.write(f"Parsed split rows add up to **€{parsed_total:.2f}**.")

        if receipt_total is not None and abs(parsed_total - receipt_total) >= 0.02:
            st.warning(
                f"Parsed rows differ from the receipt total by €{parsed_total - receipt_total:.2f}. "
                "Check the OCR and parsed items debug panel before settling."
            )

        balances = calculate_balances(
            st.session_state.receipt_items,
            st.session_state.splits,
            payer_name,
        )

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

elif st.session_state.started and len(st.session_state.receipt_items) == 0:
    st.error("No receipt items were parsed. Open 'Show OCR text' and check whether the item section was recognized.")
else:
    st.info("Add participants, upload a receipt, choose who paid, then click **Start splitting**.")