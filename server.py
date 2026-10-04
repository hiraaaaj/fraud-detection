"""Full-stack server: REST API + SQLite DB + frontend.   python server.py  ->  http://localhost:8000"""
import json, os, sqlite3, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from predict import make_row, score, explain, META

DB = "fraud.db"
def db():
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row; return c
with db() as c:
    c.execute("""CREATE TABLE IF NOT EXISTS transactions(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL,
        amount REAL, merchant TEXT, city TEXT, home_city TEXT, hour INT, device_new INT, international INT,
        probability REAL, risk TEXT, action TEXT, reasons TEXT, status TEXT DEFAULT 'open')""")

class H(BaseHTTPRequestHandler):
    def send(self, obj, code=200, ctype="application/json"):
        b = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", len(b)); self.end_headers(); self.wfile.write(b)
    def body(self): return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
    def log_message(self, *a): pass

    def do_GET(self):
        if self.path == "/": return self.send(Path("web/index.html").read_bytes(), ctype="text/html; charset=utf-8")
        if self.path.startswith("/api/transactions"):
            with db() as c: rows = [dict(r) for r in c.execute("SELECT * FROM transactions ORDER BY id DESC LIMIT 50")]
            return self.send(rows)
        if self.path == "/api/stats":
            with db() as c:
                r = c.execute("SELECT COUNT(*) n, SUM(risk!='LOW') flagged, SUM(CASE WHEN risk!='LOW' THEN amount ELSE 0 END) at_risk FROM transactions").fetchone()
                by = {x["risk"]: x["k"] for x in c.execute("SELECT risk, COUNT(*) k FROM transactions GROUP BY risk")}
            return self.send({"total": r["n"], "flagged": r["flagged"] or 0, "amount_at_risk": r["at_risk"] or 0, "by_risk": by, "model": META["best_model"]})
        self.send({"error": "not found"}, 404)

    def do_POST(self):
        try:
            if self.path == "/api/score":
                d = self.body()
                row = make_row(float(d["amount"]), float(d.get("avg_amount", 500)), int(d["hour"]), d["merchant"], d["city"], d["home_city"],
                               int(d.get("device_new", 0)), int(d.get("international", 0)), float(d.get("gap_min", 2000)) * 60, int(d.get("tx_count_24h", 0)))
                s = score(row); ex = explain(row).to_dict("records")
                with db() as c:
                    cur = c.execute("INSERT INTO transactions(ts,amount,merchant,city,home_city,hour,device_new,international,probability,risk,action,reasons) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                        (time.time(), d["amount"], d["merchant"], d["city"], d["home_city"], d["hour"], d.get("device_new", 0), d.get("international", 0), s["probability"], s["risk"], s["action"], json.dumps(ex)))
                return self.send({**s, "id": cur.lastrowid, "reasons": ex})
            if self.path.startswith("/api/review/"):   # analyst: confirm/dismiss
                tid = int(self.path.rsplit("/", 1)[1]); st = self.body().get("status")
                if st not in ("confirmed_fraud", "false_alarm", "open"): return self.send({"error": "bad status"}, 400)
                with db() as c: c.execute("UPDATE transactions SET status=? WHERE id=?", (st, tid))
                return self.send({"ok": True})
        except (KeyError, ValueError) as e:
            return self.send({"error": f"invalid input: {e}"}, 400)
        self.send({"error": "not found"}, 404)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000)); print(f"Fraud Detection running on port {port}"); ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()
