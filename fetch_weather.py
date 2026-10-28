"""Open-Meteo 과거 기상 데이터(무료, API 키 불필요)로 시간별 기온/습도를 받아 전국 평균 CSV로 저장.

사용:  python fetch_weather.py --start 2025-01-01 --end 2025-12-31 --out data/weather_2025.csv

주의
- 관측소(ASOS) 실측이 아니라 재분석(모델) 기반 격자 값임. 포트폴리오엔 이 출처를 명시할 것.
- 6개 대도시 단순 평균임 (인구/수요 가중 평균으로 바꾸면 더 현실적).
- 시간대는 Asia/Seoul 기준.
"""
import argparse
import json
import urllib.parse
import urllib.request

import pandas as pd

CITIES = {
    "서울": (37.5665, 126.9780),
    "인천": (37.4563, 126.7052),
    "대전": (36.3504, 127.3845),
    "대구": (35.8714, 128.6014),
    "광주": (35.1595, 126.8526),
    "부산": (35.1796, 129.0756),
}


def fetch_city(lat, lon, start, end):
    qs = urllib.parse.urlencode(
        {
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
            "hourly": "temperature_2m,relative_humidity_2m",
            "timezone": "Asia/Seoul",
        }
    )
    url = f"https://archive-api.open-meteo.com/v1/archive?{qs}"
    with urllib.request.urlopen(url, timeout=60) as r:
        h = json.load(r)["hourly"]
    return pd.DataFrame(
        {
            "datetime": pd.to_datetime(h["time"]),
            "temperature": h["temperature_2m"],
            "humidity": h["relative_humidity_2m"],
        }
    ).set_index("datetime")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--out", default="data/weather_2025.csv")
    args = ap.parse_args()

    frames = []
    for name, (lat, lon) in CITIES.items():
        print(f"  {name} 다운로드 중...")
        frames.append(fetch_city(lat, lon, args.start, args.end))
    avg = sum(frames) / len(frames)
    avg = avg.round(2)
    avg.to_csv(args.out)
    print(f"저장 완료: {args.out} ({len(avg)}행)")


if __name__ == "__main__":
    main()
