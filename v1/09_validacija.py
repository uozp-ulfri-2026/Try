"""
Validacija sentiment modela.

Uporaba:
  1. Odpri validacija_slepa.csv (brez modelovih napovedi!)
  2. V stolpec 'moja_ocena' vpiši: positive, negative, ali neutral
  3. Poženi: python3 09_validacija.py validacija_slepa.csv validacija_kljuc.csv

Izpiše: accuracy, Cohen's kappa, confusion matrix, napačne napovedi.
"""

import sys
import pandas as pd
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, confusion_matrix, classification_report
)

def main():
    if len(sys.argv) < 3:
        print("Uporaba: python3 09_validacija.py validacija_slepa.csv validacija_kljuc.csv")
        return

    blind = pd.read_csv(sys.argv[1])
    key = pd.read_csv(sys.argv[2])

    # Združi slepo oceno z modelovimi napovedmi
    df = blind.merge(key, on="id", how="inner")

    # Preveri da je moja_ocena izpolnjena
    df["moja_ocena"] = df["moja_ocena"].fillna("").astype(str).str.strip()
    missing = df["moja_ocena"] == ""
    if missing.any():
        print(f"⚠  Še {missing.sum()} neoznačenih vrstic. Izpolni 'moja_ocena' za vse.")
        print(df.loc[missing, ["id", "lead"]].head(5).to_string())
        return

    y_true = df["moja_ocena"].str.strip().str.lower()
    y_pred = df["sentiment_label"].str.strip().str.lower()

    labels = ["negative", "neutral", "positive"]

    # Metrike
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print("=" * 50)
    print("VALIDACIJA SENTIMENT MODELA")
    print("=" * 50)
    print(f"\nŠtevilo vzorcev:  {len(df)}")
    print(f"Accuracy:         {acc:.1%}")
    print(f"Cohen's kappa:    {kappa:.3f}")

    print(f"\nConfusion matrix (vrstice=ročno, stolpci=model):")
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df.to_string())

    print(f"\nClassification report:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    # Napačne napovedi
    wrong = df[y_true != y_pred]
    if len(wrong) > 0:
        print(f"\n{'=' * 50}")
        print(f"NAPAČNE NAPOVEDI ({len(wrong)}):")
        print(f"{'=' * 50}")
        for _, r in wrong.iterrows():
            print(f"\n  ID: {r.id}")
            print(f"  Model: {r.sentiment_label}  |  Ročno: {r.moja_ocena}")
            print(f"  Lead: {str(r.lead)[:150]}")
            if pd.notna(r.get("komentar")) and str(r.komentar).strip():
                print(f"  Komentar: {r.komentar}")
    else:
        print("\n✓ Vse napovedi pravilne!")

if __name__ == "__main__":
    main()
