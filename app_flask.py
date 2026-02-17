import os
import json
import pandas as pd
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
from werkzeug.utils import secure_filename
import traceback
import joblib
from data_processor import prepare_data_inference
model = joblib.load("xgboost_breakout_best.pkl")
scaler = joblib.load("scaler.pkl")
feature_list = joblib.load("feature_list.pkl")
# Import các hàm indicator và signal engine
from indicators import (
    compute_moving_averages,
    compute_rsi,
    compute_macd,
    compute_bollinger_bands,
    compute_volume_sma,
    compute_dmi
)
from signal_engine import generate_buy_signals

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =====================================================
# 1. LOAD + COMPUTE INDICATORS (dùng cho dashboard)
# =====================================================
def load_and_compute(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.lower()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time", "ticker"])
    df = df.sort_values("time")

    df = compute_moving_averages(df)
    df = compute_rsi(df, window=14)
    df = compute_macd(df)
    df = compute_bollinger_bands(df)
    df = compute_volume_sma(df, window=9)
    df = compute_dmi(df, window=14)

    return df

# =====================================================
# 2. VẼ BIỂU ĐỒ – TRADINGVIEW / FIREANT STYLE (giữ nguyên của bạn)
# =====================================================
def build_ml_features(df: pd.DataFrame) -> pd.DataFrame:

    df["Return_1"] = df["close"].pct_change(1)
    df["Return_5"] = df["close"].pct_change(5)

    df["BB_width"] = df["bb_upper"] - df["bb_lower"]
    df["Trend_strength"] = df["adx"]

    df["Volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    df["Position_ratio"] = (
        (df["close"] - df["bb_lower"]) /
        (df["bb_upper"] - df["bb_lower"])
    )

    df["High_Low_Range"] = df["high"] - df["low"]

    # Nếu model có RSI14 thì đảm bảo đúng tên
    if "rsi14" in df.columns:
        df["RSI14"] = df["rsi14"]

    df = df.dropna()

    return df
def create_figure(df: pd.DataFrame, ticker: str):
    # Lấy dữ liệu nến cuối cùng để hiển thị lên nhãn
    last = df.iloc[-1]
    
    # 1. TẠO LAYOUT 4 HÀNG
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.15, 0.15, 0.15], 
        vertical_spacing=0.015,
        specs=[[{"secondary_y": True}], [{}], [{}], [{}]]
    )

    # ROW 1: PRICE + VOLUME
    vol_colors = ['rgba(8, 153, 129, 0.5)' if c >= o else 'rgba(242, 54, 69, 0.5)' for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(
        x=df["time"], y=df["volume"], name="Volume", 
        marker_color=vol_colors, legendgroup="vol", showlegend=True
    ), row=1, col=1, secondary_y=True)
    
    if "volume_sma9" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["time"], y=df["volume_sma9"], name="Vol SMA9", 
            line=dict(color="#FFD600", width=1.5), legendgroup="vol", showlegend=False
        ), row=1, col=1, secondary_y=True)

    # Bollinger Bands
    if "bb_upper" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["time"], y=df["bb_upper"], name="Bollinger Bands", 
            line=dict(color="#00B0F0", width=0.8), opacity=0.7, legendgroup="bb", showlegend=True
        ), row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(
            x=df["time"], y=df["bb_lower"], name="BB Lower", 
            line=dict(color="#00B0F0", width=0.8), opacity=0.7, 
            fill='tonexty', fillcolor='rgba(0, 176, 240, 0.05)', legendgroup="bb", showlegend=False
        ), row=1, col=1, secondary_y=False)

    # Price (Nến)
    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], 
        name="Price", increasing_line_color='#089981', decreasing_line_color='#F23645', line=dict(width=1), showlegend=False
    ), row=1, col=1, secondary_y=False)

    # MA Lines
    ma_colors = {"ma10": "#F50057", "ma20": "#2962FF", "ma60": "#FF6D00"}
    for ma, color in ma_colors.items():
        if ma in df.columns:
            fig.add_trace(go.Scatter(
                x=df["time"], y=df[ma], name=ma.upper(), 
                line=dict(color=color, width=1), showlegend=True
            ), row=1, col=1, secondary_y=False)

    # ROW 2: RSI
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["rsi14"], name="RSI", 
        line=dict(color="#AB47BC", width=1.5), fill='tozeroy', fillcolor='rgba(171, 71, 188, 0.1)', legendgroup="rsi", showlegend=True
    ), row=2, col=1)
    for val in [30, 70]:
        fig.add_hline(y=val, line_dash="dash", line_width=1, line_color="#787B86", row=2, col=1)

    # ROW 3: DMI
    fig.add_trace(go.Scatter(x=df["time"], y=df["plus_di"], name="DMI", line=dict(color="#00E676", width=1), legendgroup="dmi", showlegend=True), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["minus_di"], name="-DI", line=dict(color="#FF5252", width=1), legendgroup="dmi", showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["adx"], name="ADX", line=dict(color="#FFD600", width=1.5), legendgroup="dmi", showlegend=False), row=3, col=1)

    # ROW 4: MACD
    macd_hist = df["macd_hist"].values
    hist_colors = []
    col_grow_pos, col_shrink_pos, col_grow_neg, col_shrink_neg = '#26A69A', '#B2DFDB', '#EF5350', '#FFCDD2'
    for i in range(len(macd_hist)):
        curr, prev = macd_hist[i], macd_hist[i-1] if i > 0 else 0
        if curr >= 0: hist_colors.append(col_grow_pos if curr >= prev else col_shrink_pos)
        else: hist_colors.append(col_grow_neg if curr <= prev else col_shrink_neg)

    fig.add_trace(go.Bar(
        x=df["time"], y=df["macd_hist"], name="MACD", marker_color=hist_colors, marker_line_width=0, opacity=0.9, legendgroup="macd", showlegend=True
    ), row=4, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["macd"], name="MACD Line", line=dict(color="#2962FF", width=1.5), legendgroup="macd", showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["macd_signal"], name="Signal Line", line=dict(color="#FF6D00", width=1.5), legendgroup="macd", showlegend=False), row=4, col=1)
    fig.add_hline(y=0, line_color="#555555", line_width=1, row=4, col=1)

    # Annotations (giữ nguyên)
    close_color = "#089981" if last['close'] >= last['open'] else "#F23645"
    text_r1_line1 = (f"<b>{ticker}</b> ({last['time'].strftime('%d/%m')})  "
                     f"O: {last['open']}  H: {last['high']}  L: {last['low']}  "
                     f"C: <span style='color:{close_color}'><b>{last['close']}</b></span>")
    text_r1_line2 = f"Vol: {int(last['volume']):,}"
    if "volume_sma9" in df.columns: text_r1_line2 += f"  <span style='color:#FFD600'>SMA9: {int(last['volume_sma9']):,}</span>"
    text_r1_line3 = ""
    if "bb_upper" in df.columns: text_r1_line3 += f"<span style='color:#00B0F0'>BB(20,2) {last['bb_upper']:.2f} {last['bb_lower']:.2f}</span>  "
    if "ma10" in df.columns: text_r1_line3 += f"<span style='color:#F50057'>MA10 {last['ma10']:.2f}</span>  "
    if "ma20" in df.columns: text_r1_line3 += f"<span style='color:#2962FF'>MA20 {last['ma20']:.2f}</span>  "
    if "ma60" in df.columns: text_r1_line3 += f"<span style='color:#FF6D00'>MA60 {last['ma60']:.2f}</span>"
    full_text_row1 = f"{text_r1_line1}<br>{text_r1_line2}<br>{text_r1_line3}"
    fig.add_annotation(text=full_text_row1, xref="x domain", yref="y domain", x=0.01, y=1, showarrow=False, align="left", font=dict(size=11, color="#e1e1e1"), bgcolor="rgba(0,0,0,0.5)", row=1, col=1)

    fig.add_annotation(text=f"<span style='color:#AB47BC'><b>RSI (14): {last['rsi14']:.2f}</b></span>", xref="x domain", yref="y domain", x=0.01, y=1, showarrow=False, align="left", font=dict(size=11), row=2, col=1)

    text_dmi = (f"<span style='color:#FFD600'>ADX(14): {last['adx']:.2f}</span>  "
                f"<span style='color:#00E676'>+DI: {last['plus_di']:.2f}</span>  "
                f"<span style='color:#FF5252'>-DI: {last['minus_di']:.2f}</span>")
    fig.add_annotation(text=text_dmi, xref="x domain", yref="y domain", x=0.01, y=1, showarrow=False, align="left", font=dict(size=11), row=3, col=1)

    text_macd = (f"<span style='color:#2962FF'>MACD(12,26): {last['macd']:.2f}</span>  "
                 f"<span style='color:#FF6D00'>Signal(9): {last['macd_signal']:.2f}</span>  "
                 f"Hist: {last['macd_hist']:.2f}")
    fig.add_annotation(text=text_macd, xref="x domain", yref="y domain", x=0.01, y=1, showarrow=False, align="left", font=dict(size=11), row=4, col=1)

    # Layout và menu zoom (giữ nguyên)
    layout_normal = {"yaxis.domain": [0.46, 1.0], "yaxis2.domain": [0.46, 1.0], "yaxis3.domain": [0.31, 0.45], "yaxis4.domain": [0.16, 0.30], "yaxis5.domain": [0.00, 0.15]}
    layout_focus_price = {"yaxis.domain": [0.19, 1.0], "yaxis2.domain": [0.19, 1.0], "yaxis3.domain": [0.13, 0.18], "yaxis4.domain": [0.07, 0.12], "yaxis5.domain": [0.00, 0.06]}
    layout_focus_rsi = {"yaxis.domain": [0.80, 1.0], "yaxis2.domain": [0.80, 1.0], "yaxis3.domain": [0.15, 0.79], "yaxis4.domain": [0.08, 0.14], "yaxis5.domain": [0.00, 0.07]}
    layout_focus_dmi = {"yaxis.domain": [0.80, 1.0], "yaxis2.domain": [0.80, 1.0], "yaxis3.domain": [0.73, 0.79], "yaxis4.domain": [0.15, 0.72], "yaxis5.domain": [0.00, 0.14]}
    layout_focus_macd = {"yaxis.domain": [0.80, 1.0], "yaxis2.domain": [0.80, 1.0], "yaxis3.domain": [0.73, 0.79], "yaxis4.domain": [0.66, 0.72], "yaxis5.domain": [0.00, 0.65]}

    updatemenus = [dict(
        type="buttons", direction="left", x=0.0, y=1.07, xanchor="left", yanchor="top", pad={"r": 10, "t": 10}, showactive=True,
        bgcolor="#455A64", bordercolor="#90A4AE", borderwidth=1, font=dict(color="#FFFFFF", size=12, family="Arial, sans-serif"),
        buttons=[
            dict(label="Mặc định", method="relayout", args=[layout_normal]),
            dict(label="🔍 Giá", method="relayout", args=[layout_focus_price]),
            dict(label="🔍 RSI", method="relayout", args=[layout_focus_rsi]),
            dict(label="🔍 DMI", method="relayout", args=[layout_focus_dmi]),
            dict(label="🔍 MACD", method="relayout", args=[layout_focus_macd])
        ]
    )]

    tick_vals = df["time"]
    tick_text = [f"<b>T{t.month}/{str(t.year)[2:]}</b>" if t.day == 1 else t.strftime('%d/%m') for t in df["time"]]

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#131722", paper_bgcolor="#131722",
        margin=dict(l=0, r=60, t=30, b=10),
        height=1000, hovermode="x unified",
        updatemenus=updatemenus,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)", font=dict(size=11, color="#ccc"), itemclick="toggle", itemdoubleclick="toggleothers"),
        xaxis=dict(showspikes=True, spikemode="across", spikesnap="cursor", showline=False, spikedash="dash", spikecolor="#555"),
        yaxis=dict(showspikes=True, spikemode="across", spikesnap="cursor", showline=False, spikedash="dash", spikecolor="#555"),
    )

    fig.update_xaxes(type='category', tickmode='array', tickvals=tick_vals, ticktext=tick_text, nticks=10, gridcolor="#2a2e39", griddash="dot", rangebreaks=[dict(bounds=["sat", "mon"])], rangeslider_visible=False)
    fig.update_yaxes(gridcolor="#2a2e39", griddash="dot", side="right", showspikes=True, row=1, col=1, secondary_y=False)
    max_vol = df["volume"].max() if not df["volume"].empty else 1
    fig.update_yaxes(range=[0, max_vol * 4], showgrid=False, showticklabels=False, secondary_y=True, row=1, col=1)
    fig.update_yaxes(range=[0, 100], tickvals=[30, 70], gridcolor="#2a2e39", griddash="dot", side="right", row=2, col=1)
    fig.update_yaxes(gridcolor="#2a2e39", griddash="dot", side="right", row=3, col=1)
    fig.update_yaxes(gridcolor="#2a2e39", griddash="dot", side="right", row=4, col=1)

    # =========================
    # AI BREAKOUT MARKER
    # =========================
    if "ai_prediction" in df.columns:
        breakout_points = df[df["ai_prediction"] == 1]

        fig.add_trace(go.Scatter(
            x=breakout_points["time"],
            y=breakout_points["low"] * 0.995,
            mode="markers",   # 🔥 bỏ +text
            marker=dict(
                symbol="triangle-up",
                size=14,
                color="#00FF99",
                line=dict(color="white", width=2)
            ),
            name="AI Breakout"
        ), row=1, col=1)

    # 🔥 QUAN TRỌNG: return phải nằm ngoài if
    return fig

# =====================================================
# 4. ROUTES
# =====================================================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            return redirect(url_for("dashboard", filename=filename))
    return render_template("index.html")


@app.route("/dashboard/<filename>")
def dashboard(filename):

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    df_all = load_and_compute(filepath)

    tickers = sorted(df_all["ticker"].unique())
    selected_ticker = request.args.get("ticker", tickers[0])
    start_date = request.args.get("start")
    end_date = request.args.get("end")
    df_display = df_all[df_all["ticker"] == selected_ticker].copy()

    # =========================
    # FILTER THEO DATE
    # =========================
    if start_date:
        df_display = df_display[df_display["time"] >= pd.to_datetime(start_date)]

    if end_date:
        df_display = df_display[df_display["time"] <= pd.to_datetime(end_date)]
    # =========================
    # GIỚI HẠN SỐ NẾN RENDER (TĂNG TỐC)
    # =========================
    if not start_date and not end_date:
        df_display = df_display.tail(500)
    if df_display.empty:
        return render_template(
            "dashboard.html",
            graphJSON=None,
            tickers=tickers,
            selected_ticker=selected_ticker,
            filename=filename,
            table_html="<p>Không có dữ liệu trong khoảng thời gian này.</p>",
            last_close=None,
            last_open=None,
            last_high=None,
            last_low=None,
            last_change=0,
            last_change_pct=0
        )
    # ===== AI BREAKOUT PREDICTION =====
    try:
        # 1️⃣ Build đúng feature giống lúc train
        df_features = build_ml_features(df_display.copy())

        # 2️⃣ One-hot ngành
        if "industry_level2" in df_features.columns:
            df_features["industry_level2"] = df_features["industry_level2"].fillna("Unknown")
            df_features = pd.get_dummies(
                df_features,
                columns=["industry_level2"],
                prefix="Sector"
            )

        # 3️⃣ Đảm bảo đủ feature
        for col in feature_list:
            if col not in df_features.columns:
                df_features[col] = 0

        X_input = df_features[feature_list]

        # 4️⃣ Scale
        X_scaled = scaler.transform(X_input)

        # 5️⃣ Predict
        THRESHOLD = 0.7
        y_prob = model.predict_proba(X_scaled)[:, 1]

        df_features["ai_prediction"] = (y_prob >= THRESHOLD).astype(int)

        print("Max prob:", y_prob.max())
        print("AI prediction sum:", df_features["ai_prediction"].sum())

        # 6️⃣ Merge lại vào df_display
        df_display = df_display.merge(
            df_features[["time", "ai_prediction"]],
            on="time",
            how="left"
        )

    except Exception as e:
        print("AI error:", e)
        df_display["ai_prediction"] = 0
    # =========================
    # HEADER VALUES
    # =========================
    last_close = last_open = last_high = last_low = None
    last_change = 0
    last_change_pct = 0

    if not df_display.empty:
        last_row = df_display.iloc[-1]

        last_close = round(last_row.get("close", 0), 2)
        last_open  = round(last_row.get("open", 0), 2)
        last_high  = round(last_row.get("high", 0), 2)
        last_low   = round(last_row.get("low", 0), 2)

        if len(df_display) >= 2:
            prev_close = df_display.iloc[-2].get("close", last_close)
            last_change = round(last_close - prev_close, 2)

            if prev_close != 0:
                last_change_pct = round((last_close / prev_close - 1) * 100, 2)

    # =========================
    # CREATE FIGURE
    # =========================
    fig = create_figure(df_display, selected_ticker)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    table_html = df_display.sort_values("time", ascending=False).to_html(
        classes="table-data", index=False, float_format="%.2f"
    )

    return render_template(
        "dashboard.html",
        graphJSON=graphJSON,
        tickers=tickers,
        selected_ticker=selected_ticker,
        filename=filename,
        table_html=table_html,
        last_close=last_close,
        last_open=last_open,
        last_high=last_high,
        last_low=last_low,
        last_change=last_change,
        last_change_pct=last_change_pct
    )
@app.route("/download/<filename>/<ticker>")
def download_data(filename, ticker):
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    df_all = load_and_compute(filepath)
    df = df_all[df_all["ticker"] == ticker]
    return Response(
        df.to_csv(index=False),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={ticker}_analysis.csv"}
    )


print(feature_list)
if __name__ == "__main__":
    app.run()