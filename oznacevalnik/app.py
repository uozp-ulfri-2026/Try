"""
Oznacevalnik sentimenta — Flask spletna aplikacija.

Lokalni zagon:
  python3 app.py

Docker: nastavi env spremenljivke DB_PATH, PORT, SECRET (opcijsko).
Izvoz: /izvoz
"""

import csv
import io
import os
import sqlite3
from functools import wraps
from flask import Flask, render_template, redirect, url_for, request, Response

DB     = os.environ.get("DB_PATH", os.path.join(os.path.dirname(__file__), "articles.db"))
PORT   = int(os.environ.get("PORT", 5050))
SECRET = os.environ.get("SECRET", "")   # ce je prazen, ni avtentikacije

app = Flask(__name__)


def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not SECRET:
            return f(*args, **kwargs)
        auth = request.authorization
        if not auth or auth.password != SECRET:
            return Response(
                "Dostop zavrnjen.",
                401,
                {"WWW-Authenticate": 'Basic realm="Oznacevalnik"'},
            )
        return f(*args, **kwargs)
    return decorated


def get_db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def stats(conn):
    total   = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    done    = conn.execute("SELECT COUNT(*) FROM labels WHERE label != 'skip'").fetchone()[0]
    skipped = conn.execute("SELECT COUNT(*) FROM labels WHERE label = 'skip'").fetchone()[0]
    return total, done, skipped


@app.route("/")
@require_auth
def index():
    return redirect(url_for("oznaci"))


@app.route("/oznaci")
@require_auth
def oznaci():
    conn = get_db()
    article = conn.execute("""
        SELECT a.* FROM articles a
        LEFT JOIN labels l ON a.id = l.article_id
        WHERE l.article_id IS NULL
        ORDER BY a.sort_order
        LIMIT 1
    """).fetchone()

    total, done, skipped = stats(conn)
    conn.close()

    if article is None:
        return redirect(url_for("konec"))

    pct = int(100 * done / total) if total else 0
    return render_template("index.html",
                           article=article,
                           done=done, total=total,
                           skipped=skipped, pct=pct)


@app.route("/oznaci/<article_id>", methods=["POST"])
@require_auth
def shrani(article_id):
    label = request.form.get("label", "")
    if label in ("positive", "negative", "neutral", "skip"):
        conn = get_db()
        conn.execute(
            "INSERT OR REPLACE INTO labels (article_id, label) VALUES (?, ?)",
            (article_id, label),
        )
        conn.commit()
        conn.close()
    return redirect(url_for("oznaci"))


@app.route("/razveljavi", methods=["POST"])
@require_auth
def razveljavi():
    conn = get_db()
    last = conn.execute(
        "SELECT article_id FROM labels ORDER BY labeled_at DESC LIMIT 1"
    ).fetchone()
    if last:
        conn.execute("DELETE FROM labels WHERE article_id = ?", (last["article_id"],))
        conn.commit()
    conn.close()
    return redirect(url_for("oznaci"))


@app.route("/konec")
@require_auth
def konec():
    conn = get_db()
    total, done, skipped = stats(conn)
    conn.close()
    return render_template("done.html", total=total, done=done, skipped=skipped)


@app.route("/izvoz")
@require_auth
def izvoz():
    conn = get_db()
    rows = conn.execute("""
        SELECT a.id, a.url, a.date, a.title, a.lead, a.topics, l.label, l.labeled_at
        FROM labels l
        JOIN articles a ON a.id = l.article_id
        WHERE l.label != 'skip'
        ORDER BY l.labeled_at
    """).fetchall()
    conn.close()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "url", "date", "title", "lead", "topics", "label", "labeled_at"])
    for row in rows:
        writer.writerow(list(row))

    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=oznake.csv"},
    )


if __name__ == "__main__":
    if not os.path.exists(DB):
        print(f"NAPAKA: {DB} ne obstaja. Najprej pozeni: python3 prepare.py")
        raise SystemExit(1)
    app.run(host="0.0.0.0", port=PORT, debug=False)
