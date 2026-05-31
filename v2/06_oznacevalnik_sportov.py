"""
06_oznacevalnik_sportov.py

Terminal app za ročno popravljanje klasifikacije športov.
Prikazuje članke urejene po zaupanju (najnegotovejši najprej).
Oznake se shranijo v sport_labels.csv za self-training.

Uporaba:
  python3 06_oznacevalnik_sportov.py
  python3 06_oznacevalnik_sportov.py --max-conf 0.5   # samo negotovi
  python3 06_oznacevalnik_sportov.py --sport Rokomet  # samo en šport
  python3 06_oznacevalnik_sportov.py --all            # vsi (ne samo negotovi)
"""

import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd

# ANSI
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

TARGET_SPORTS = [
    "Nogomet", "Rokomet", "Alpsko smučanje", "Kolesarstvo", "Košarka",
    "Hokej na ledu", "Atletika", "Smučarski skoki", "Odbojka",
    "Športno plezanje", "Biatlon", "Tenis",
]
ALL_CHOICES = TARGET_SPORTS + ["Drugo"]

INPUT  = "sport_clanki.parquet"
OUTPUT = "sport_labels.csv"


def conf_color(c):
    if c < 0.35:
        return RED
    if c < 0.50:
        return YELLOW
    return GREEN


def print_separator():
    print(f"{DIM}{'─' * 60}{RESET}")


def print_article(pos, total, row):
    c = float(row.get("confidence", 0))
    sport = str(row["sport"])

    print_separator()
    conf_str = f"{conf_color(c)}{c:.3f}{RESET}"

    gap_str = ""
    if "conf_gap" in row and pd.notna(row.get("conf_gap")):
        gap = float(row["conf_gap"])
        gap_color = RED if gap < 0.05 else YELLOW if gap < 0.12 else GREEN
        gap_str = f"  gap {gap_color}{gap:.3f}{RESET}"

    print(f"{BOLD}[{pos}/{total}]{RESET}  {BOLD}{sport}{RESET}  conf {conf_str}{gap_str}\n")

    title = str(row.get("title", "") or "")
    print(f"{BOLD}{title}{RESET}\n")

    lead = str(row.get("lead", "") or "")[:280]
    if lead:
        print(f"{DIM}{lead}...{RESET}\n")

    # Top alternativni sporti
    if "top2_sport" in row and pd.notna(row.get("top2_sport")):
        alts = []
        alts.append(f"  1▸ {BOLD}{sport:22s}{RESET} {c:.3f}")
        alts.append(f"  2  {str(row['top2_sport']):22s} {float(row.get('top2_conf',0)):.3f}")
        if "top3_sport" in row and pd.notna(row.get("top3_sport")):
            alts.append(f"  3  {str(row['top3_sport']):22s} {float(row.get('top3_conf',0)):.3f}")
        print("\n".join(alts) + "\n")


def print_menu(current_sport):
    cols = 3
    rows_needed = (len(ALL_CHOICES) + cols - 1) // cols
    for r in range(rows_needed):
        line = ""
        for c in range(cols):
            idx = r + c * rows_needed
            if idx < len(ALL_CHOICES):
                n = idx + 1
                s = ALL_CHOICES[idx]
                marker = BOLD if s == current_sport else ""
                end    = RESET if s == current_sport else ""
                line += f"  {DIM}{n:2d}.{RESET} {marker}{s:22s}{end}"
        print(line)
    print()
    print(f"[enter] Potrdi '{BOLD}{current_sport}{RESET}'   "
          f"[1-{len(ALL_CHOICES)}] Popravi   [s] Skip   [u] Undo   [q] Shrani in izhod")


def load_labeled_ids(path):
    if not os.path.exists(path):
        return set()
    try:
        df = pd.read_csv(path)
        return set(df["id"].astype(str))
    except Exception:
        return set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",    default=INPUT)
    parser.add_argument("--output",   default=OUTPUT)
    parser.add_argument("--sport",    default=None,  help="Filtriraj po športu")
    parser.add_argument("--max-conf", type=float, default=0.55,
                        help="Prikaži samo članke z confidence ≤ X (privzeto 0.55)")
    parser.add_argument("--all",      action="store_true",
                        help="Prikaži vse članke, ne samo negotove")
    args = parser.parse_args()

    print(f"Berem {args.input}...")
    df = pd.read_parquet(args.input)

    # Filtriraj na naše 12 športe + Drugo
    df = df[df["sport"].isin(TARGET_SPORTS + ["Drugo"])].copy()

    if args.sport:
        df = df[df["sport"] == args.sport]

    if not args.all:
        df = df[df["confidence"] <= args.max_conf]

    # Sortiraj po conf_gap (ascending) ali po confidence
    if "conf_gap" in df.columns:
        df = df.sort_values("conf_gap", ascending=True)
    else:
        df = df.sort_values("confidence", ascending=True)

    # Preskoči že označene
    labeled_ids = load_labeled_ids(args.output)
    df = df[~df["id"].astype(str).isin(labeled_ids)]

    total = len(df)
    print(f"Za označevanje: {total} člankov")
    if total == 0:
        print("Ni nič za označit.")
        return

    # Odpri CSV za append
    file_exists = os.path.exists(args.output)
    fout = open(args.output, "a", newline="", encoding="utf-8")
    writer = csv.writer(fout)
    if not file_exists:
        writer.writerow(["id", "sport", "original_sport", "confidence", "conf_gap"])

    labeled = skipped = 0
    history = []  # [(csv_row_id, article_index_in_df)]
    rows_list = list(df.iterrows())
    pos = 0
    offset = len(labeled_ids)  # že označeni iz prejšnjih sej

    while pos < len(rows_list):
        _, row = rows_list[pos]
        print_article(offset + pos + 1, offset + total, row)
        print_menu(str(row["sport"]))

        while True:
            try:
                choice = input("> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                choice = "q"

            if choice == "q":
                fout.close()
                print(f"\nOznačenih: {labeled}  preskočenih: {skipped}")
                print(f"Shranjeno: {args.output}")
                return

            if choice == "u":
                if not history:
                    print(f"  Ni kaj razveljaviti")
                    continue
                last_id, last_pos = history.pop()
                _undo(args.output, last_id)
                labeled -= 1
                pos = last_pos
                print(f"{YELLOW}  Razveljavljeno{RESET}")
                break  # ponovi artikel na last_pos

            if choice == "s":
                skipped += 1
                history.append((None, pos))
                pos += 1
                break

            if choice == "":
                _save(writer, fout, row, str(row["sport"]))
                history.append((str(row["id"]), pos))
                labeled += 1
                pos += 1
                break

            try:
                n = int(choice)
                if 1 <= n <= len(ALL_CHOICES):
                    chosen = ALL_CHOICES[n - 1]
                    orig   = str(row["sport"])
                    _save(writer, fout, row, chosen)
                    history.append((str(row["id"]), pos))
                    labeled += 1
                    pos += 1
                    if chosen != orig:
                        print(f"{YELLOW}  {orig} → {chosen}{RESET}")
                    break
                else:
                    print(f"  Vnesi število med 1 in {len(ALL_CHOICES)}")
            except ValueError:
                print(f"  Neveljaven vnos")

    fout.close()
    print(f"\nKonec. Označenih: {labeled}  preskočenih: {skipped}")
    print(f"Shranjeno: {args.output}")


def _undo(output_path, article_id):
    if not os.path.exists(output_path):
        return
    df = pd.read_csv(output_path)
    df = df[df["id"].astype(str) != str(article_id)]
    df.to_csv(output_path, index=False)


def _save(writer, fout, row, sport):
    writer.writerow([
        str(row["id"]),
        sport,
        str(row["sport"]),
        round(float(row.get("confidence", 0)), 4),
        round(float(row.get("conf_gap", 0) or 0), 4),
    ])
    fout.flush()


if __name__ == "__main__":
    main()
