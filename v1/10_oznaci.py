"""
Interaktivno označevanje sentimenta.

Uporaba:
  python3 10_oznaci.py validacija_slepa.csv

Prikaže lead enega po enega. Pritisni:
  p = positive, n = negative, u = neutral, s = preskoči, q = shrani & končaj

Napredek se sproti shranjuje v CSV.
"""

import sys
import pandas as pd

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "validacija_slepa.csv"
    df = pd.read_csv(path)
    df["moja_ocena"] = df["moja_ocena"].fillna("").astype(str).str.strip()

    todo = df[df["moja_ocena"] == ""].index.tolist()
    done = len(df) - len(todo)

    if not todo:
        print("✓ Vse vrstice so že označene!")
        return

    print(f"Označenih: {done}/{len(df)}. Še {len(todo)}.")
    print("Tipke: [p]ositive  [n]egative  ne[u]tral  [s]preskoči  [q]shrani & končaj")
    print("=" * 60)

    mapping = {"p": "positive", "n": "negative", "u": "neutral"}

    for i, idx in enumerate(todo, 1):
        row = df.loc[idx]
        print(f"\n[{done + i}/{len(df)}]  ID: {int(row.id)}  |  {row.date}")
        print(f"  {row.lead}")
        print()

        while True:
            choice = input("  > ").strip().lower()
            if choice in mapping:
                df.at[idx, "moja_ocena"] = mapping[choice]
                df.to_csv(path, index=False)
                break
            elif choice == "s":
                break
            elif choice == "q":
                df.to_csv(path, index=False)
                done_now = (df["moja_ocena"] != "").sum()
                print(f"\nShranjeno. Označenih: {done_now}/{len(df)}.")
                return
            else:
                print("  ? p/n/u/s/q")

    print(f"\n✓ Končano! Vseh {len(df)} označenih.")
    df.to_csv(path, index=False)

if __name__ == "__main__":
    main()
