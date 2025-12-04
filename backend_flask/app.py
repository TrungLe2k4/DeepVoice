# backend_flask/app.py
from flask import Flask, request, jsonify, redirect
from flask_cors import CORS
import os
import json

from api import (
    MODEL_PATH,
    N_FEAT,
    scaler_stats,
    res2net_metrics,
    LOG_PATH,
    load_fast_model,
    analyze_features,
    get_last_event,
)

app = Flask(__name__)
CORS(app)

# Nạp model 1 lần khi khởi động (có cache bên trong)
_model, has_model = load_fast_model()

# =========================
# 0️⃣  ROOT: redirect / → /dashboard (đỡ 404)
# =========================
@app.route("/", methods=["GET"])
def index():
    return redirect("/dashboard")


# =========================
# 1️⃣  API: /health
# =========================
@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "ok": True,
            "model_file": os.path.basename(MODEL_PATH),
            "has_model": bool(has_model),
            "n_features": N_FEAT,
            "scaler_stats": scaler_stats,
            "res2net_metrics": res2net_metrics,
            "version": "dv-1.0.0",
        }
    )


# =========================
# 2️⃣  API: /analyze
# =========================
@app.route("/analyze", methods=["POST"])
def analyze_endpoint():
    try:
        data = request.get_json(force=True) or {}
        feats = data.get("features", None)

        if feats is None:
            return (
                jsonify({"error": "Thiếu trường 'features' trong JSON"}),
                400,
            )

        result = analyze_features(feats)
        return jsonify(result)

    except ValueError as e:
        # lỗi validate features
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print("[ANALYZE_ERROR]", e)
        return jsonify({"error": str(e)}), 500


# =========================
# 3️⃣  API BACKEND REALTIME: /status
# =========================
@app.route("/status", methods=["GET"])
def status():
    """
    Trả về event phân tích mới nhất để UI backend hiển thị.
    """
    return jsonify(get_last_event())


# =========================
# 4️⃣  API LỊCH SỬ: /events (JSON)
# =========================
@app.route("/events", methods=["GET"])
def events():
    """
    Trả về danh sách các event gần nhất (JSON) để frontend /history dùng.
    Query param: ?limit=100 (default 50)
    """
    limit = request.args.get("limit", default=50, type=int)
    limit = max(1, min(limit, 1000))

    rows = []
    if os.path.exists(LOG_PATH):
        try:
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines[-limit:]:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        except Exception as e:
            print("[EVENTS_READ_ERR]", e)

    rows.sort(key=lambda x: x.get("ts", ""), reverse=True)
    return jsonify(rows)


# =========================
# 5️⃣  DASHBOARD REALTIME: /dashboard
# =========================
@app.route("/dashboard", methods=["GET"])
def dashboard():
    html = """
    <!doctype html>
    <html lang="vi">
    <head>
      <meta charset="utf-8">
      <title>DeepVoice Guard – Backend Monitor</title>
      <style>
        body {
          font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background: #f5f5f7;
          margin: 0;
          padding: 24px;
        }
        .card {
          max-width: 520px;
          margin: 0 auto;
          background: #fff;
          border-radius: 16px;
          box-shadow: 0 10px 30px rgba(0,0,0,0.08);
          padding: 20px 24px 24px;
        }
        .title {
          font-size: 20px;
          font-weight: 600;
          margin-bottom: 4px;
        }
        .sub {
          font-size: 13px;
          color: #666;
          margin-bottom: 16px;
        }
        .dot {
          width: 12px;
          height: 12px;
          border-radius: 999px;
          margin-right: 8px;
        }
        .row {
          display: flex;
          align-items: center;
          margin-bottom: 8px;
        }
        .meter {
          position: relative;
          height: 10px;
          border-radius: 999px;
          background: #e5e5ea;
          overflow: hidden;
          margin: 8px 0 4px;
        }
        .meter-fill {
          position: absolute;
          inset: 0;
          width: 0%;
          background: linear-gradient(90deg, #34c759, #ff3b30);
          transition: width 0.25s ease-out;
        }
        .label-row {
          display: flex;
          justify-content: space-between;
          font-size: 12px;
          color: #555;
          margin-bottom: 8px;
        }
        .reason {
          font-size: 13px;
          color: #333;
          margin-top: 8px;
          white-space: pre-wrap;
        }
        .flags {
          font-size: 12px;
          color: #555;
          margin-top: 4px;
        }
        .chip {
          display: inline-flex;
          align-items: center;
          padding: 2px 8px;
          border-radius: 999px;
          background: #f2f2f7;
          font-size: 11px;
          margin-right: 4px;
          margin-top: 4px;
        }
        .chip span {
          font-size: 10px;
          margin-right: 4px;
        }
        .meta {
          font-size: 11px;
          color: #888;
          margin-top: 8px;
        }
        .link-row {
          margin-top: 12px;
          font-size: 12px;
        }
        .link-row a {
          color: #007bff;
          text-decoration: none;
        }
        .link-row a:hover {
          text-decoration: underline;
        }
      </style>
    </head>
    <body>
      <div class="card">
        <div class="row">
          <div id="dot" class="dot" style="background:#34c759;"></div>
          <div>
            <div class="title">DeepVoice Guard – Backend Monitor</div>
            <div class="sub">Theo dõi các lần gọi /analyze từ Chrome Extension</div>
          </div>
        </div>

        <div class="label-row">
          <div>Xác suất giả mạo (prob_fused)</div>
          <div id="prob-label">0.000</div>
        </div>
        <div class="meter">
          <div id="meter-fill" class="meter-fill"></div>
        </div>

        <div class="label-row">
          <div>Level: <span id="level">green</span></div>
          <div>SNR: <span id="snr">0.0</span> dB</div>
        </div>

        <div class="reason" id="reasons">Chưa có dữ liệu. Hãy mở Google Meet và bật extension.</div>
        <div class="flags" id="flags"></div>
        <div class="meta" id="ts"></div>

        <div class="link-row">
          Xem lịch sử chi tiết: <a href="/history" target="_blank">/history</a>
        </div>
      </div>

      <script>
        function updateUI(data) {
          const prob = Number(data.prob_fused || 0);
          const level = data.level || "green";
          const snr = Number(data.snr || 0);
          const flags = data.flags || {};
          const reasons = data.reasons || [];
          const ts = data.ts || "";

          const fill = document.getElementById("meter-fill");
          const probLabel = document.getElementById("prob-label");
          const levelEl = document.getElementById("level");
          const snrEl = document.getElementById("snr");
          const dot = document.getElementById("dot");
          const reasonEl = document.getElementById("reasons");
          const flagsEl = document.getElementById("flags");
          const tsEl = document.getElementById("ts");

          const p = Math.max(0, Math.min(1, prob));
          fill.style.width = (p * 100).toFixed(1) + "%";
          probLabel.textContent = p.toFixed(3);

          levelEl.textContent = level;
          snrEl.textContent = snr.toFixed(1);

          if (level === "red") {
            dot.style.background = "#ff3b30";
          } else if (level === "amber") {
            dot.style.background = "#ff9500";
          } else {
            dot.style.background = "#34c759";
          }

          if (reasons.length > 0) {
            reasonEl.textContent = "• " + reasons.join("\\n• ");
          } else {
            reasonEl.textContent = "Không có lý do chi tiết (reasons trống).";
          }

          const flagKeys = Object.keys(flags).filter(k => flags[k]);
          if (flagKeys.length > 0) {
            flagsEl.innerHTML = flagKeys.map(k =>
              "<span class='chip'><span>⚑</span>" + k + "</span>"
            ).join(" ");
          } else {
            flagsEl.textContent = "";
          }

          tsEl.textContent = ts ? ("Last event: " + ts) : "";
        }

        async function poll() {
          try {
            const res = await fetch("/status");
            if (!res.ok) throw new Error("HTTP " + res.status);
            const data = await res.json();
            updateUI(data);
          } catch (e) {
            console.error(e);
          }
        }

        poll();
        setInterval(poll, 1000);
      </script>
    </body>
    </html>
    """
    return html


# =========================
# 6️⃣  LỊCH SỬ ĐẸP: /history
# =========================
@app.route("/history", methods=["GET"])
def history_page():
    html = """
    <!doctype html>
    <html lang="vi">
    <head>
      <meta charset="utf-8">
      <title>DeepVoice Guard – History</title>
      <style>
        body {
          font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background: #f5f5f7;
          margin: 0;
          padding: 24px;
        }
        .container {
          max-width: 920px;
          margin: 0 auto;
          background: #fff;
          border-radius: 16px;
          box-shadow: 0 10px 30px rgba(0,0,0,0.08);
          padding: 20px 24px 24px;
        }
        h1 {
          font-size: 20px;
          margin-top: 0;
          margin-bottom: 4px;
        }
        .sub {
          font-size: 13px;
          color: #666;
          margin-bottom: 16px;
        }
        table {
          width: 100%;
          border-collapse: collapse;
          font-size: 12px;
        }
        th, td {
          border-bottom: 1px solid #eee;
          padding: 6px 8px;
          text-align: left;
          vertical-align: top;
        }
        th {
          background: #f9f9fb;
          font-weight: 600;
        }
        tr:nth-child(even) td {
          background: #fafafa;
        }
        .badge {
          display: inline-block;
          padding: 2px 8px;
          border-radius: 999px;
          font-size: 11px;
          color: #fff;
        }
        .badge.green { background: #34c759; }
        .badge.amber { background: #ff9500; }
        .badge.red { background: #ff3b30; }
        .flags {
          font-size: 11px;
          color: #555;
        }
        .flag-chip {
          display: inline-block;
          padding: 1px 6px;
          border-radius: 999px;
          background: #f2f2f7;
          margin-right: 4px;
          margin-top: 2px;
        }
        .reasons {
          white-space: pre-wrap;
        }
        .toolbar {
          margin-bottom: 12px;
          display: flex;
          justify-content: space-between;
          align-items: center;
          font-size: 12px;
        }
        select, input {
          font-size: 12px;
          padding: 3px 6px;
          border-radius: 8px;
          border: 1px solid #ccc;
          outline: none;
        }
      </style>
    </head>
    <body>
      <div class="container">
        <h1>DeepVoice Guard – Lịch sử phân tích</h1>
        <div class="sub">Đọc từ Logs/events.jsonl (backend_flask/Logs/events.jsonl)</div>

        <div class="toolbar">
          <div>
            Hiển thị:
            <select id="limit">
              <option value="20">20</option>
              <option value="50" selected>50</option>
              <option value="100">100</option>
              <option value="200">200</option>
            </select>
            bản ghi mới nhất
          </div>
          <div>
            Bộ lọc level:
            <select id="filter-level">
              <option value="">Tất cả</option>
              <option value="green">green</option>
              <option value="amber">amber</option>
              <option value="red">red</option>
            </select>
          </div>
        </div>

        <table>
          <thead>
            <tr>
              <th>Thời gian (UTC)</th>
              <th>Prob</th>
              <th>Level</th>
              <th>SNR (dB)</th>
              <th>Flags</th>
              <th>Reasons</th>
            </tr>
          </thead>
          <tbody id="tbody">
            <tr><td colspan="6">Đang tải dữ liệu...</td></tr>
          </tbody>
        </table>
      </div>

      <script>
        async function loadData() {
          const limit = document.getElementById("limit").value;
          const filterLevel = document.getElementById("filter-level").value;
          const tbody = document.getElementById("tbody");
          tbody.innerHTML = "<tr><td colspan='6'>Đang tải dữ liệu...</td></tr>";

          try {
            const res = await fetch("/events?limit=" + encodeURIComponent(limit));
            if (!res.ok) throw new Error("HTTP " + res.status);
            let data = await res.json();

            if (filterLevel) {
              data = data.filter(row => row.level === filterLevel);
            }

            if (!data.length) {
              tbody.innerHTML = "<tr><td colspan='6'>Không có dữ liệu.</td></tr>";
              return;
            }

            const rowsHtml = data.map(ev => {
              const ts = ev.ts || "";
              const prob = Number(ev.prob_fused || 0).toFixed(3);
              const level = ev.level || "green";
              const snr = Number(ev.snr || 0).toFixed(1);
              const reasons = (ev.reasons || []).map(r => "• " + r).join("\\n");
              const flags = ev.flags || {};
              const flagKeys = Object.keys(flags).filter(k => flags[k]);

              let badgeClass = "green";
              if (level === "red") badgeClass = "red";
              else if (level === "amber") badgeClass = "amber";

              const flagsHtml = flagKeys.length
                ? flagKeys.map(k => "<span class='flag-chip'>" + k + "</span>").join(" ")
                : "";

              return `
                <tr>
                  <td>${ts}</td>
                  <td>${prob}</td>
                  <td><span class="badge ${badgeClass}">${level}</span></td>
                  <td>${snr}</td>
                  <td class="flags">${flagsHtml}</td>
                  <td class="reasons">${reasons}</td>
                </tr>
              `;
            }).join("");

            tbody.innerHTML = rowsHtml;
          } catch (e) {
            console.error(e);
            tbody.innerHTML = "<tr><td colspan='6'>Lỗi tải dữ liệu.</td></tr>";
          }
        }

        document.getElementById("limit").addEventListener("change", loadData);
        document.getElementById("filter-level").addEventListener("change", loadData);

        loadData();
      </script>
    </body>
    </html>
    """
    return html


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
