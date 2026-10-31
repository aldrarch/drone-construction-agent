"""
Streamlit UI for the drone construction agent.

Russian-language interface for monitoring individual residential construction
from drone photos.
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image

from config import MODEL_TYPE, CLIP_MODEL_NAME, REPORTS_DIR
from model_interface import create_model
from schemas.prediction import (
    GPSStatus,
    ImageAnalysisResult,
    ObjectSummary,
    StageLabel,
    RiskFlag,
)
from services.analysis_service import analyze_single_image, summarize
from services.report_service import build_report, save_report

st.set_page_config(page_title="AI-агент мониторинга ИЖС", layout="wide")

st.title("AI-агент мониторинга строительства ИЖС")
st.write(
    "Приложение предварительно оценивает стадию строительства индивидуального жилого дома "
    "по фотографиям с дрона и формирует JSON-отчёт. "
    "Используется CLIP ViT-B/32 в режиме zero-shot классификации."
)

# ---------- Sidebar: model / device info ----------
with st.sidebar:
    st.header("Информация о модели")
    try:
        model = create_model(None)
        model.load()
        device_info = (
            f"Тип модели: {model.model_type}\n"
            f"Модель: {model.model_name}\n"
            f"Устройство: {model._device}"
        )
        st.code(device_info, language="text")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Не удалось загрузить модель: {exc}")
        st.stop()

st.divider()

# ---------- Block 1: Photos upload ----------
st.header("1. Загрузка фотографий")
uploaded_files = st.file_uploader(
    "Загрузите одну или несколько фотографий объекта (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_files:
    cols = st.columns(min(6, len(uploaded_files)))
    for idx, file in enumerate(uploaded_files):
        col = cols[idx % len(cols)]
        try:
            img = Image.open(file)
            col.image(img, caption=file.name, use_container_width=True)
        except Exception:  # noqa: BLE001
            col.text(file.name)
    st.success(f"Загружено файлов: {len(uploaded_files)}")

st.divider()

# ---------- Block 2: Object info ----------
st.header("2. Информация об объекте")

col1, col2 = st.columns(2)
with col1:
    project_id = st.text_input("ID объекта", value="OBJECT_001")
    expected_stage_raw = st.selectbox(
        "Ожидаемая стадия строительства",
        options=[s.value for s in StageLabel],
        index=1,
    )
    expected_stage = StageLabel(expected_stage_raw)

with col2:
    object_lat = st.number_input(
        "Широта объекта (WGS-84)",
        value=55.0,
        format="%.6f",
    )
    object_lon = st.number_input(
        "Долгота объекта (WGS-84)",
        value=37.0,
        format="%.6f",
    )
    comment = st.text_area("Комментарий (необязательно)", height=100)

st.divider()

# ---------- Block 3: Analysis button ----------
st.header("3. Анализ")
analyze_clicked = st.button("Проанализировать объект", type="primary", use_container_width=True)

results: list[ImageAnalysisResult] = []
summary: ObjectSummary | None = None

if analyze_clicked:
    if not uploaded_files:
        st.warning("Сначала загрузите фотографии.")
    else:
        for file in uploaded_files:
            with st.spinner(f"Обработка {file.name}..."):
                try:
                    # Save uploaded content to a temp path so PIL can reopen it.
                    tmp_path = Path("data/uploads") / file.name
                    tmp_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp_path.write_bytes(file.getvalue())
                    result = analyze_single_image(
                        str(tmp_path),
                        model=model,
                        expected_stage=expected_stage,
                        object_latitude=object_lat if object_lat else None,
                        object_longitude=object_lon if object_lon else None,
                    )
                    results.append(result)
                except Exception as exc:  # noqa: BLE001
                    results.append(
                        ImageAnalysisResult(
                            filename=file.name,
                            predicted_stage=StageLabel.STAGE_0,
                            scores={},
                            score_gap=0.0,
                            confidence=0.0,
                            manual_review_required=True,
                            gps_status=GPSStatus.GPS_NOT_AVAILABLE,
                            error=str(exc),
                        )
                    )
        summary = summarize(results, expected_stage)


# ---------- Block 4: Results per photo ----------
if results:
    st.divider()
    st.header("4. Результаты по фотографиям")

    for res in results:
        st.subheader(res.filename)
        if res.error:
            st.error(f"Ошибка: {res.error}")

        pred = res.predicted_stage
        cols = st.columns(2)
        cols[0].metric("Предполагаемая стадия", pred.value)
        cols[1].metric("Уверенность", f"{res.confidence:.3f}")

        st.write("Оценки сходства:")
        score_cols = st.columns(3)
        for idx, stage in enumerate([StageLabel.STAGE_0, StageLabel.STAGE_50, StageLabel.STAGE_100]):
            score = res.scores.get(stage.value, 0.0)
            score_cols[idx].metric(stage.value, f"{score:.3f}")

        st.write("Разница между первым и вторым результатом:", f"{res.score_gap:.3f}")

        gps_label = res.gps_status.value
        gps_cols = st.columns(2)
        gps_cols[0].write(f"GPS-статус: {gps_label}")
        if res.gps_status == GPSStatus.GPS_FOUND:
            gps_cols[1].write(
                f"Координаты фото: {res.gps_latitude:.6f}, {res.gps_longitude:.6f}"
            )
            if res.object_latitude is not None and res.object_longitude is not None:
                gps_cols[1].write(
                    f"Координаты объекта: {res.object_latitude:.6f}, {res.object_longitude:.6f}"
                )
                gps_cols[1].write(f"Расстояние: {res.distance_km} км")
        elif res.gps_status == GPSStatus.GPS_NOT_AVAILABLE:
            gps_cols[1].write("GPS-координаты в фотографии не найдены.")

        if res.manual_review_required:
            st.warning("Требуется ручная проверка")
        else:
            st.success("Ручная проверка не требуется")

    st.divider()

    # ---------- Block 5: Object summary ----------
    st.header("5. Итог по объекту")
    if summary:
        summ_cols = st.columns(4)
        summ_cols[0].metric("Обработано фото", summary.processed_images)
        summ_cols[1].metric("Фото с GPS", summary.images_with_gps)
        summ_cols[2].metric("Требуют проверки", summary.images_needing_review)
        summ_cols[3].metric("Общий риск", summary.risk_status.value)

        st.write("Объяснение результата:")
        st.info(summary.explanation)

        if summary.contradictory_predictions:
            st.warning(
                "Фотографии показывают разные стадии строительства. "
                "Не усредняйте результат — требуется ручная проверка."
            )

    st.divider()

    # ---------- Block 6: Download report ----------
    st.header("6. Отчёт")
    if summary and results:
        report = build_report(
            project_id=project_id,
            model_type=MODEL_TYPE,
            model_name=CLIP_MODEL_NAME,
            summary=summary,
            images=results,
        )
        path = save_report(report, REPORTS_DIR)
        json_bytes = report.model_dump_json(indent=2).encode("utf-8")
        st.download_button(
            label="Скачать JSON-отчёт",
            data=json_bytes,
            file_name=path.name,
            mime="application/json",
        )
        st.code(report.model_dump_json(indent=2), language="json")
