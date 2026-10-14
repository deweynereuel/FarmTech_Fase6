"""Detecção de chair e bottle em tempo real com a webcam do PC.

O programa carrega o best.pt produzido na Entrega 1, processa a webcam local e
salva evidências auditáveis quando o operador pressiona S. Não realiza treino.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time
import types
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = PROJECT_DIR.parent
DEFAULT_WEIGHTS = REPOSITORY_DIR / "runs" / "treino_60ep" / "weights" / "best.pt"
DEFAULT_EVIDENCE_DIR = PROJECT_DIR / "evidencias"


def import_runtime_dependencies() -> None:
    """Importa as bibliotecas pesadas apenas ao iniciar a detecção.

    Assim, ``python webcam_yolo.py --help`` continua disponível antes da
    instalação, e a mensagem de erro para a execução normal fica objetiva.
    """
    global cv2, torch
    try:
        import cv2 as opencv
        import torch as pytorch
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Dependência ausente. Ative o ambiente virtual e execute: "
            "pip install -r ir_alem_webcam/requirements.txt"
        ) from error
    cv2, torch = opencv, pytorch


def install_pathlib_checkpoint_compatibility() -> None:
    """Compatibiliza pesos salvos por Python mais novo em Python 3.12.

    Alguns pesos do YOLOv5 guardam objetos ``Path`` dentro do checkpoint.
    Versões recentes do Python os serializam como ``pathlib._local.*``; esse
    submódulo não existe no Python 3.12 do ambiente Windows. O alias abaixo é
    usado somente na desserialização e converte o caminho para o tipo Windows.
    """
    if "pathlib._local" in sys.modules:
        return
    compatibility_module = types.ModuleType("pathlib._local")
    compatibility_module.Path = pathlib.WindowsPath
    compatibility_module.PosixPath = pathlib.WindowsPath
    compatibility_module.WindowsPath = pathlib.WindowsPath
    sys.modules["pathlib._local"] = compatibility_module
    setattr(pathlib, "_local", compatibility_module)


def parse_args() -> argparse.Namespace:
    """Lê os parâmetros para tornar a demonstração reproduzível."""
    parser = argparse.ArgumentParser(
        description="Executa o modelo YOLOv5 customizado na webcam do PC."
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                        help="Caminho para o best.pt da Entrega 1.")
    parser.add_argument("--camera", type=int, default=0,
                        help="Índice da webcam (0 = câmera padrão).")
    parser.add_argument("--backend", choices=("auto", "dshow", "msmf", "any"),
                        default="auto", help="Backend do OpenCV para abrir a câmera.")
    parser.add_argument("--width", type=int, default=1280,
                        help="Largura solicitada à webcam.")
    parser.add_argument("--height", type=int, default=720,
                        help="Altura solicitada à webcam.")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Tamanho de entrada do YOLOv5.")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confiança mínima para desenhar uma detecção.")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="Limiar de IoU da supressão de caixas duplicadas.")
    parser.add_argument("--device", default="", help="Dispositivo PyTorch: vazio, cpu ou 0.")
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_EVIDENCE_DIR,
                        help="Pasta que receberá imagens e metadados das capturas.")
    parser.add_argument("--yolov5-dir", type=Path,
                        help="Clone local do YOLOv5; evita download pelo torch.hub.")
    return parser.parse_args()


def camera_backend(name: str) -> int | None:
    """Mapeia a opção legível para a constante de captura do OpenCV."""
    backends: dict[str, int | None] = {
        "auto": cv2.CAP_DSHOW if sys.platform.startswith("win") else None,
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
        "any": None,
    }
    return backends[name]


def open_camera(args: argparse.Namespace) -> cv2.VideoCapture:
    """Abre a webcam e solicita a resolução escolhida pelo operador."""
    backend = camera_backend(args.backend)
    capture = cv2.VideoCapture(args.camera) if backend is None else cv2.VideoCapture(args.camera, backend)
    if not capture.isOpened():
        raise RuntimeError(
            f"Não foi possível abrir a câmera {args.camera} com backend '{args.backend}'. "
            "Feche outros programas que usam a câmera ou teste --camera 1 / --backend msmf."
        )
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    return capture


def load_model(args: argparse.Namespace) -> Any:
    """Carrega o peso customizado, via clone local ou cache oficial do PyTorch."""
    if not args.weights.is_file():
        raise FileNotFoundError(
            f"Peso não encontrado: {args.weights}\n"
            "Confirme se a pasta runs/ da Entrega 1 está no mesmo repositório ou use --weights."
        )

    install_pathlib_checkpoint_compatibility()

    if args.yolov5_dir:
        if not (args.yolov5_dir / "hubconf.py").is_file():
            raise FileNotFoundError("--yolov5-dir deve apontar para um clone válido do YOLOv5.")
        model = torch.hub.load(
            str(args.yolov5_dir), "custom", path=str(args.weights), source="local", device=args.device
        )
    else:
        # O torch.hub baixa/cacheia o código oficial do YOLOv5 apenas na primeira execução.
        model = torch.hub.load(
            "ultralytics/yolov5", "custom", path=str(args.weights), device=args.device
        )
    model.conf = args.conf
    model.iou = args.iou
    model.max_det = 50
    return model


def draw_detections(frame: Any, detections: Any, names: Any) -> tuple[Any, list[dict[str, Any]]]:
    """Desenha as caixas e devolve dados serializáveis para as evidências."""
    annotated = frame.copy()
    records: list[dict[str, Any]] = []
    palette = ((41, 196, 255), (81, 201, 84), (228, 113, 52), (194, 73, 245))

    for x1, y1, x2, y2, confidence, class_id in detections.cpu().numpy():
        cid = int(class_id)
        class_name = names[cid] if isinstance(names, (list, tuple)) else names[cid]
        color = palette[cid % len(palette)]
        left, top, right, bottom = (round(value) for value in (x1, y1, x2, y2))
        label = f"{class_name} {confidence:.2f}"
        cv2.rectangle(annotated, (left, top), (right, bottom), color, 2)
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_top = max(top, text_height + baseline + 8)
        cv2.rectangle(annotated, (left, label_top - text_height - baseline - 6),
                      (left + text_width + 8, label_top), color, -1)
        cv2.putText(annotated, label, (left + 4, label_top - baseline - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA)
        records.append({
            "class_id": cid, "class_name": str(class_name), "confidence": round(float(confidence), 4),
            "x1": left, "y1": top, "x2": right, "y2": bottom,
        })
    return annotated, records


def save_evidence(save_dir: Path, raw: Any, annotated: Any, detections: list[dict[str, Any]],
                  fps: float, args: argparse.Namespace) -> None:
    """Salva imagens e metadados reais de uma captura realizada pelo operador."""
    save_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    raw_name = f"raw_{stamp}.jpg"
    detected_name = f"detected_{stamp}.jpg"
    metadata_name = f"capture_{stamp}.json"
    cv2.imwrite(str(save_dir / raw_name), raw)
    cv2.imwrite(str(save_dir / detected_name), annotated)

    metadata = {
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "weights": str(args.weights.resolve()), "camera_index": args.camera,
        "confidence_threshold": args.conf, "inference_image_size": args.imgsz,
        "estimated_fps": round(fps, 2), "raw_image": raw_name,
        "annotated_image": detected_name, "detections": detections,
    }
    with (save_dir / metadata_name).open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    csv_path = save_dir / "detections.csv"
    write_header = not csv_path.exists()
    fields = ["captured_at", "raw_image", "annotated_image", "estimated_fps",
              "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"]
    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for detection in detections:
            writer.writerow({
                "captured_at": metadata["captured_at"], "raw_image": raw_name,
                "annotated_image": detected_name, "estimated_fps": metadata["estimated_fps"], **detection,
            })
    print(f"Evidência salva em: {save_dir / detected_name}")


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf deve estar entre 0 e 1.")
    import_runtime_dependencies()
    model = load_model(args)
    capture = open_camera(args)
    frame_times: deque[float] = deque(maxlen=20)
    print("Webcam iniciada. Pressione S para salvar evidência; Q ou Esc para encerrar.")

    try:
        while True:
            start = time.perf_counter()
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("A webcam deixou de fornecer frames.")

            # A chamada recebe o frame BGR do OpenCV e o AutoShape faz o pré/pós-processamento.
            results = model(frame, size=args.imgsz)
            annotated, detections = draw_detections(frame, results.xyxy[0], model.names)
            elapsed = time.perf_counter() - start
            frame_times.append(elapsed)
            fps = len(frame_times) / sum(frame_times) if sum(frame_times) else 0.0
            status = f"FPS: {fps:.1f} | deteccoes: {len(detections)} | S: salvar | Q: sair"
            cv2.rectangle(annotated, (0, 0), (min(annotated.shape[1], 520), 36), (25, 25, 25), -1)
            cv2.putText(annotated, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("FarmTech | YOLOv5 webcam", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("s"), ord("S")):
                save_evidence(args.save_dir, frame, annotated, detections, fps, args)
            if key in (ord("q"), ord("Q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
        print("Recursos da webcam liberados.")


if __name__ == "__main__":
    main()
