from pathlib import Path
import pandas as pd
import json
from pyproj import Transformer


def main():
    BASE = Path(__file__).resolve().parent
    DATA = BASE / "data"
    OUT = BASE / "outputs"
    OUT.mkdir(exist_ok=True)

    # 專門給前端/靜態站點用的輸出資料夾
    PUBLIC = OUT / "public"
    PUBLIC.mkdir(exist_ok=True)

    pred_path = OUT / "pred_next_hour_advanced.csv"
    cent_path = DATA / "taxi_zone_centroids.csv"

    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    if not cent_path.exists():
        raise FileNotFoundError(cent_path)

    # 讀資料（driver view 只需要 PULocationID + pred_rides）
    df_pred = pd.read_csv(pred_path)
    if "PULocationID" not in df_pred.columns or "pred_rides" not in df_pred.columns:
        raise ValueError("pred_next_hour_advanced.csv 必須包含 PULocationID 與 pred_rides 欄位")

    df_cent = pd.read_csv(cent_path)  # 需要：LocationID, lon, lat, Borough, Zone

    # 座標 2263 → 4326
    transformer = Transformer.from_crs("EPSG:2263", "EPSG:4326", always_xy=True)
    lon_wgs, lat_wgs = transformer.transform(df_cent["lon"].values, df_cent["lat"].values)
    df_cent["lon_wgs"] = lon_wgs.round(6)
    df_cent["lat_wgs"] = lat_wgs.round(6)

    df_cent = df_cent[["LocationID", "Borough", "Zone", "lat_wgs", "lon_wgs"]]

    df = df_pred.merge(
        df_cent,
        left_on="PULocationID",
        right_on="LocationID",
        how="left"
    )

    df = df.dropna(subset=["lat_wgs", "lon_wgs"])
    df["pred_rides"] = pd.to_numeric(df["pred_rides"], errors="coerce").fillna(0.0).round(3)

    df = df[["PULocationID", "pred_rides", "Borough", "Zone", "lat_wgs", "lon_wgs"]]

    # ===== 給前端使用的 JSON（可直接 fetch）=====
    zones_records = df.to_dict(orient="records")
    zones_json = json.dumps(zones_records, ensure_ascii=False)
    center_lat = float(df["lat_wgs"].mean())
    center_lon = float(df["lon_wgs"].mean())

    (PUBLIC / "zones.json").write_text(json.dumps(zones_records, ensure_ascii=False, indent=2), encoding="utf-8")
    (PUBLIC / "meta.json").write_text(json.dumps({"center_lat": center_lat, "center_lon": center_lon}, ensure_ascii=False, indent=2), encoding="utf-8")

    html = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>司機派車建議（Demo）</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<style>
  html, body {
    margin:0;
    height:100%;
    font-family:Arial,"微軟正黑體",sans-serif;
  }
  #app {
    display:flex;
    flex-direction:row;
    height:100vh;
    width:100vw;
  }
  #map { flex:2; }
  #panel {
    flex:1;
    border-left:1px solid #ccc;
    padding:10px;
    box-sizing:border-box;
    overflow:auto;
  }
  .zone-card {
    padding:10px;
    border:1px solid #ddd;
    border-radius:8px;
    margin-bottom:10px;
    cursor:pointer;
    font-size:13px;
  }
  .zone-card:hover { background:#f5f5f5; }
  .zone-card.selected {
    border-color:#ff9800;
    box-shadow:0 0 6px rgba(255,152,0,0.7);
  }
  .top1 { background:#ffe0b2; }
  .top2 { background:#e1f5fe; }
  .top3 { background:#e8f5e9; }
</style>
</head>

<body>
<div id="app">
  <div id="map"></div>

  <div id="panel">
    <h2>司機派車建議</h2>
    <p style="font-size:13px;">
      操作方式：<br>
      1. 在左邊地圖點一下設定司機位置<br>
      2. 系統會列出推薦前 3 名區域<br>
      3. 點任一張卡片，小車會沿道路線跑過去
    </p>

    <h3 id="driver-pos" style="font-size:14px;">尚未選擇司機位置</h3>
    <h3>推薦前 3 名區域</h3>
    <div id="rank"></div>

    <h3>說明</h3>
    <p style="font-size:12px; color:#555; line-height:1.5;">
      • <b>需求分數</b>：使用模型預測的下一小時需求量 (pred_rides)。<br>
      • <b>距離</b>：依 OSRM 公開 API 取得的道路距離 (公里)。<br>
      • <b>預估車資</b>：示意用途，假設「起跳 $2.5 + 每公里 $1.5」。<br>
      • <b>綜合分數</b>：
        <code>score = 1.0 × 需求 + 0.3 × 車資 − 0.5 × 距離</code>（可依營運需求調整）。<br>
      • <b>小車動畫</b>：點擊某推薦卡片後，車子會沿著路線行駛至該區域。
    </p>

    <p style="font-size:11px; color:#777;">
      ※ 本頁為 Demo UI，僅示範派車決策的視覺化呈現，實際派車邏輯可依營運調整。
    </p>
  </div>
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

<script>
// ===== Python 產生的資料（已內嵌，也另外輸出 zones.json 給前端 fetch 用）=====
var zones = __ZONES_JSON__;

// ===== 地圖初始化 =====
var map = L.map('map').setView([__CENTER_LAT__, __CENTER_LON__], 11);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom:19
}).addTo(map);

// ===== 車子 icon =====
var carIcon = L.icon({
  iconUrl: './taxi.jpg',
  iconSize: [42,42],
  iconAnchor: [21,21]
});

// ===== 畫需求氣泡 =====
zones.forEach(function(z){
  var color =
    z.pred_rides > 0.8 ? '#ff0000' :
    z.pred_rides > 0.5 ? '#ff9800' :
    z.pred_rides > 0.1 ? '#ffff00' :
                        '#00c853';

  L.circleMarker([z.lat_wgs, z.lon_wgs], {
    radius: Math.min(35, z.pred_rides * 2 + 5),
    color: color,
    fillColor: color,
    fillOpacity: 0.6
  }).bindTooltip(
    "Zone：" + z.Zone + "<br>需求：" + z.pred_rides,
    {sticky:true}
  ).addTo(map);
});

// ===== 直線距離（公里） =====
function haversine(lat1, lon1, lat2, lon2){
  var R = 6371;
  var dLat = (lat2-lat1)*Math.PI/180;
  var dLon = (lon2-lon1)*Math.PI/180;
  var a =
    Math.sin(dLat/2)**2 +
    Math.cos(lat1*Math.PI/180)*Math.cos(lat2*Math.PI/180) *
    Math.sin(dLon/2)**2;
  return R * (2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a)));
}

// ===== 預估車資（簡化版） =====
function estimateFare(dist_km){
  var base = 2.5;
  var perKm = 1.5;
  return base + perKm * Math.max(1, dist_km);
}

// ===== 綜合分數：需求 + 車資 - 距離 =====
function calcScore(pred, dist_km, fare){
  return pred * 1.0 + fare * 0.3 - dist_km * 0.5;
}

// ===== OSRM 路線 =====
async function getRoute(fromLat, fromLon, toLat, toLon){
  var url = `https://router.project-osrm.org/route/v1/driving/${fromLon},${fromLat};${toLon},${toLat}?overview=full&geometries=geojson`;
  try {
    var res = await fetch(url);
    var data = await res.json();
    if (!data.routes || !data.routes.length){
      return {coords: [], dist: null};
    }
    var r = data.routes[0];
    return {
      coords: r.geometry.coordinates.map(c => [c[1], c[0]]),
      dist: r.distance / 1000.0
    };
  } catch(e){
    return {coords: [], dist: null};
  }
}

// ===== 小車動畫 =====
var carMarker = null;
var carTimer = null;

function playCar(route){
  if (carTimer){
    clearInterval(carTimer);
    carTimer = null;
  }
  if (carMarker){
    map.removeLayer(carMarker);
    carMarker = null;
  }
  if (!route.coords || route.coords.length < 2) return;

  carMarker = L.marker(route.coords[0], {icon: carIcon}).addTo(map);
  map.fitBounds(L.latLngBounds(route.coords), {padding:[30,30]});

  var i = 0;
  carTimer = setInterval(function(){
    i++;
    if (i >= route.coords.length){
      clearInterval(carTimer);
      carTimer = null;
      return;
    }
    carMarker.setLatLng(route.coords[i]);
  }, 80);
}

// 存目前路線的 polyline
var routeLayers = [];

// ===== 點地圖：計算推薦 Top3 =====
map.on('click', async function(e){
  var lat = e.latlng.lat;
  var lon = e.latlng.lng;

  document.getElementById("driver-pos").innerHTML =
    "司機位置：<br>lat=" + lat.toFixed(5) + "<br>lon=" + lon.toFixed(5);

  // 清掉舊路線
  routeLayers.forEach(function(l){ map.removeLayer(l); });
  routeLayers = [];

  // 先用直線距離粗排
  var initial = zones.map(function(z){
    var d = haversine(lat, lon, z.lat_wgs, z.lon_wgs);
    var f = estimateFare(d);
    return {
      info: z,
      straightDist: d,
      approxScore: calcScore(z.pred_rides, d, f)
    };
  }).sort(function(a,b){ return b.approxScore - a.approxScore; }).slice(0,3);

  // 取得路線並重新計分
  var top3 = [];
  for (let item of initial){
    var z = item.info;
    var routeInfo = await getRoute(lat, lon, z.lat_wgs, z.lon_wgs);
    var dist = routeInfo.dist != null ? routeInfo.dist : item.straightDist;
    var f = estimateFare(dist);

    top3.push({
      info: z,
      coords: (routeInfo.coords && routeInfo.coords.length)
                ? routeInfo.coords
                : [[lat,lon],[z.lat_wgs,z.lon_wgs]],
      dist_km: dist,
      fare: f,
      score: calcScore(z.pred_rides, dist, f)
    });
  }

  top3.sort(function(a,b){ return b.score - a.score; });

  // 畫三條路線
  var colors = ["blue","green","purple"];
  top3.forEach(function(t, idx){
    var poly = L.polyline(t.coords, {
      color: colors[idx] || "blue",
      weight: 5,
      opacity: 0.8
    }).addTo(map);
    routeLayers.push(poly);
  });

  // 更新右邊卡片
  var rankDiv = document.getElementById("rank");
  rankDiv.innerHTML = "";

  top3.forEach(function(t, idx){
    var z = t.info;
    var card = document.createElement("div");
    card.className = "zone-card top" + (idx+1);

    card.innerHTML =
      "<b>第 " + (idx+1) + " 名</b><br>" +
      "Zone：" + z.Zone + " (ID:" + z.PULocationID + ")<br>" +
      "預測需求：" + z.pred_rides + "<br>" +
      "距離：" + t.dist_km.toFixed(2) + " km<br>" +
      "預估車資：約 $" + t.fare.toFixed(1) + "<br>" +
      "綜合分數：" + t.score.toFixed(2);

    card.onclick = function(){ playCar(t); };
    rankDiv.appendChild(card);
  });
});
</script>

</body>
</html>
"""

    html = html.replace("__ZONES_JSON__", zones_json)
    html = html.replace("__CENTER_LAT__", str(center_lat))
    html = html.replace("__CENTER_LON__", str(center_lon))

    out_path = PUBLIC / "driver_view_simple.html"
    out_path.write_text(html, encoding="utf-8")

    # taxi.jpg 也放到 public，前端可以直接連
    taxi_src = BASE / "taxi.jpg"
    if taxi_src.exists():
        (PUBLIC / "taxi.jpg").write_bytes(taxi_src.read_bytes())

    # 給你一個最直覺的入口檔名（index.html）
    (PUBLIC / "index.html").write_text(html, encoding="utf-8")

    print("✅ 已產生（可直接給前端連結/靜態部署）：")
    print(" -", out_path)
    print(" -", PUBLIC / "index.html")
    print(" -", PUBLIC / "zones.json")
    print(" -", PUBLIC / "meta.json")


if __name__ == "__main__":
    main()
